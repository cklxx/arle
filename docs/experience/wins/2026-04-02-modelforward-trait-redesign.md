# 2026-04-02 · ModelForward trait redesign: explicit phases + typed DecodeContext

## Context

Cross-model architecture review (Claude Opus + OpenAI Codex) identified two P0 issues
in the `ModelForward` trait:

1. `forward()` used `tokens.len() == 1` convention to distinguish prefill vs decode —
   implicit, error-prone, leaks into offload/graph/spec-decode logic.
2. `Box<dyn Any + Send>` for decode buffers required 6 `downcast_mut` calls — runtime
   type unsafety hidden behind `expect("type mismatch")`.

## What Worked

**Explicit phase split**: `forward_prefill(&[u32])` + `forward_decode(u32)`. The decode
method takes a bare `u32` (not a slice) — type-level guarantee of single token. The old
`forward()` remains as a convenience default that dispatches by token count, so no
external API breakage.

**Associated type `DecodeContext`**: Replaced `Box<dyn Any + Send>` with
`type DecodeContext: Send` on the trait. Scheduler field changed from
`Option<Box<dyn Any>>` to `Option<M::DecodeContext>`. Factory method
`create_decode_context(max_batch, pool)` handles lazy init in scheduler.

**Blast radius**: 10 files changed, 0 `downcast_mut` remaining, 0 `.forward(` remaining.
Both Qwen3 and Qwen3.5 implementations updated. All tests, benchmarks, examples migrated.

## Rule

When a trait method serves two distinct phases with different performance characteristics,
make them separate methods. Use associated types over `dyn Any` — the generic complexity
is worth the compile-time safety, especially on hot paths.
