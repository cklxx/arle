"""MLX inference backend for Apple Silicon.

Wraps mlx-lm's load() / stream_generate() with:
- Resource pre-flight checks (memory, disk, platform)
- Proper sampler / logits-processor construction
- Stop-sequence enforcement
- Async streaming via executor + asyncio.Queue
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from agent_infer.resource_guard import (
    check_apple_silicon,
    check_disk,
    check_memory,
    check_model_fits,
    periodic_memory_check,
)

logger = logging.getLogger(__name__)

# Default HuggingFace hub cache location
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0   # 0 = greedy (argmax)
    top_p: float = 1.0
    top_k: int = 0             # 0 = disabled (mlx-lm convention)
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str          # "stop" | "length"
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MLXEngine:
    """Inference engine backed by mlx-lm on Apple Silicon.

    Usage::

        engine = MLXEngine("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        engine.load()

        # blocking
        result = engine.generate("Hello, world!", SamplingParams(max_tokens=64))

        # async streaming
        async for text, is_final, usage in engine.generate_stream(...):
            ...
    """

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, max_memory_gb: float | None = None) -> None:
        """Load model with resource pre-checks.

        Args:
            max_memory_gb: Hard cap (GB). If the loaded model exceeds this
                value a warning is emitted (not a hard error — MLX manages
                its own memory on the Metal GPU heap).
        """
        check_apple_silicon()
        check_memory()
        check_disk(_HF_CACHE)

        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm not installed. Run:  pip install mlx-lm"
            ) from exc

        logger.info("Loading MLX model: %s", self._model_id)
        t0 = time.perf_counter()
        self._model, self._tokenizer = mlx_load(self._model_id)
        elapsed = time.perf_counter() - t0

        import mlx.core as mx
        peak_gb = mx.get_peak_memory() / (1024 ** 3)

        if max_memory_gb is not None and peak_gb > max_memory_gb:
            raise RuntimeError(
                f"Model peak memory {peak_gb:.2f} GB exceeds cap {max_memory_gb:.2f} GB."
            )

        check_model_fits(peak_gb)
        logger.info(
            "Model loaded in %.1f s — peak Metal memory %.2f GB",
            elapsed, peak_gb,
        )
        self._loaded = True

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

    # ------------------------------------------------------------------
    # Blocking generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, params: SamplingParams) -> GenerateResult:
        """Blocking (synchronous) generation. Returns the full output."""
        self._assert_loaded()
        from mlx_lm import stream_generate

        sampler, logits_processors = _build_sampling(params)
        kwargs: dict = {"sampler": sampler}
        if logits_processors:
            kwargs["logits_processors"] = logits_processors

        accumulated = ""
        last = None
        step = 0
        stop_hit = False

        for response in stream_generate(
            self._model, self._tokenizer, prompt,
            max_tokens=params.max_tokens, **kwargs,
        ):
            accumulated += response.text
            last = response
            step += 1
            periodic_memory_check(step)

            if params.stop:
                trunc, stop_hit = _check_stop(accumulated, params.stop)
                if stop_hit:
                    accumulated = trunc
                    break

        if last is None:
            return GenerateResult(
                text="",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="stop",
                prompt_tps=0.0,
                generation_tps=0.0,
                peak_memory_gb=0.0,
            )

        finish = "stop" if stop_hit else (last.finish_reason or "stop")
        return GenerateResult(
            text=accumulated,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.generation_tokens,
            finish_reason=finish,
            prompt_tps=last.prompt_tps,
            generation_tps=last.generation_tps,
            peak_memory_gb=last.peak_memory,
        )

    # ------------------------------------------------------------------
    # Async streaming generation
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> AsyncIterator[tuple[str, bool, dict | None]]:
        """Async streaming generation.

        Yields ``(token_text, is_final, usage_dict)`` tuples.

        * ``is_final`` is True only on the last item.
        * ``usage_dict`` is None on all items except the last.
        """
        self._assert_loaded()

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_sync() -> None:
            try:
                from mlx_lm import stream_generate

                sampler, logits_processors = _build_sampling(params)
                kwargs: dict = {"sampler": sampler}
                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                accumulated = ""
                last = None
                step = 0
                stop_hit = False

                for response in stream_generate(
                    self._model, self._tokenizer, prompt,
                    max_tokens=params.max_tokens, **kwargs,
                ):
                    accumulated += response.text
                    last = response
                    step += 1
                    periodic_memory_check(step)

                    if params.stop:
                        trunc, stop_hit = _check_stop(accumulated, params.stop)
                        if stop_hit:
                            # Emit only the new delta up to the stop point.
                            # The previous delta was already sent in the last
                            # iteration, so we compute the remaining piece.
                            prev_len = len(accumulated) - len(response.text)
                            new_text = trunc[prev_len:]
                            usage = _make_usage(last, stop_hit=True)
                            loop.call_soon_threadsafe(
                                queue.put_nowait, (new_text, True, usage)
                            )
                            return

                    is_final = response.finish_reason is not None
                    usage = _make_usage(last) if is_final else None
                    loop.call_soon_threadsafe(
                        queue.put_nowait, (response.text, is_final, usage)
                    )

                # If the loop ended without a finish_reason (e.g. empty model
                # output), emit a synthetic final item.
                if last is not None and last.finish_reason is None:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("", True, _make_usage(last))
                    )

            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        executor_future = loop.run_in_executor(None, _run_sync)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # (text, is_final, usage_or_None)

        await executor_future


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sampling(params: SamplingParams):
    """Return (sampler_fn, logits_processors) for the given SamplingParams."""
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    sampler = make_sampler(
        temp=params.temperature,
        top_p=params.top_p if params.top_p < 1.0 else 0.0,
        top_k=params.top_k,
    )

    logits_processors = []
    if params.repetition_penalty != 1.0:
        logits_processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty
        )

    return sampler, logits_processors


def _check_stop(text: str, stop_seqs: list[str]) -> tuple[str, bool]:
    """Return (truncated_text, hit) where hit=True if a stop seq was found."""
    for seq in stop_seqs:
        idx = text.find(seq)
        if idx != -1:
            return text[:idx], True
    return text, False


def _make_usage(response, *, stop_hit: bool = False) -> dict:
    finish = "stop" if stop_hit else (response.finish_reason or "stop")
    return {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.generation_tokens,
        "total_tokens": response.prompt_tokens + response.generation_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
        "finish_reason": finish,
    }
