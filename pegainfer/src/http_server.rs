mod openai_v1;

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::sampler::SamplingParams;
use crate::scheduler::{IncomingRequest, SchedulerHandle};
use crate::server_engine::{CompleteOutput, FinishReason, StreamDelta, Usage};
use openai_v1::{CompletionRequest, CompletionResponse, StreamChunk, StreamUsageChunk};

struct AppState {
    handle: SchedulerHandle,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, StatusCode> {
    let max_tokens = req.max_tokens_or_default();
    let stream = req.stream_or_default();
    let include_usage = req.include_usage_or_default();
    let model_id = state.handle.model_id().to_string();

    if req.prompt.trim().is_empty() {
        warn!("Rejecting empty prompt request");
        return Err(StatusCode::BAD_REQUEST);
    }

    info!(
        "Received request: prompt_len={}, max_tokens={}, stream={}",
        req.prompt.len(),
        max_tokens,
        stream,
    );

    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();

    let incoming = IncomingRequest {
        prompt: req.prompt,
        max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_k: req.top_k.unwrap_or(-1),
            top_p: req.top_p.unwrap_or(1.0),
            ignore_eos: req.ignore_eos.unwrap_or(false),
        },
        stop: req.stop,
        delta_tx,
    };

    if !state.handle.submit(incoming) {
        error!("Scheduler unavailable");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    if stream {
        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let created = now_secs();

        let stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
            let usage = delta.usage;
            let is_terminal = delta.finish_reason.is_some();

            let chunk = StreamChunk::from_delta(&request_id, created, &model_id, delta);
            let mut events = vec![Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&chunk).unwrap()),
            )];

            if include_usage
                && is_terminal
                && let Some(usage) = usage
            {
                let usage_chunk =
                    StreamUsageChunk::from_usage(&request_id, created, &model_id, usage);
                events.push(Ok(
                    Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
                ));
            }

            stream::iter(events)
        });

        let done_stream =
            stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

        Ok(Sse::new(stream.chain(done_stream)).into_response())
    } else {
        // Non-streaming: collect all deltas into a single response.
        let mut rx = delta_rx;
        let mut text = String::new();
        let mut finish_reason = FinishReason::Length;
        let mut usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        while let Some(delta) = rx.recv().await {
            text.push_str(&delta.text_delta);
            if let Some(reason) = delta.finish_reason {
                finish_reason = reason;
            }
            if let Some(u) = delta.usage {
                usage = u;
            }
        }

        info!(
            "Request completed: prompt_tokens={}, completion_tokens={}",
            usage.prompt_tokens, usage.completion_tokens
        );

        let output = CompleteOutput {
            text,
            finish_reason,
            usage,
        };
        let response = CompletionResponse::from_output(model_id, now_secs(), output);
        Ok(Json(response).into_response())
    }
}

pub fn build_app(handle: SchedulerHandle) -> Router {
    let state = Arc::new(AppState { handle });

    Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::util::ServiceExt;

    fn mock_scheduler(model_id: &str) -> SchedulerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IncomingRequest>();

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let _ = req.delta_tx.send(StreamDelta {
                    text_delta: format!("ok:{}", req.prompt),
                    finish_reason: None,
                    usage: None,
                });
                let _ = req.delta_tx.send(StreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(Usage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                });
            }
        });

        SchedulerHandle::from_parts(tx, model_id)
    }

    #[tokio::test]
    async fn completion_response_uses_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["choices"][0]["text"], "ok:hello");
    }

    #[tokio::test]
    async fn streaming_response_uses_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""model":"Qwen3-8B""#),
            "payload={payload}"
        );
        assert!(
            !payload.contains(r#""model":"qwen3-4b""#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"));
    }

    #[tokio::test]
    async fn streaming_response_includes_usage_when_requested() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload
                .contains(r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_rejects_empty_prompt() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"   ","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
