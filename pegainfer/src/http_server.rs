mod openai_v1;

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::metrics::ServerMetrics;
use crate::sampler::sampling_params_from_request;
use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerHandle};
#[cfg(test)]
use crate::server_engine::StreamDelta;
use crate::server_engine::{CompleteOutput, FinishReason, Usage};
use openai_v1::{
    ChatCompletionRequest, ChatCompletionResponse, ChatStreamChunk, ChatStreamUsageChunk,
    CompletionRequest, CompletionResponse, StreamChunk, StreamUsageChunk,
};

struct AppState {
    handle: SchedulerHandle,
    metrics: ServerMetrics,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
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
        sampling: sampling_params_from_request(
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.repetition_penalty,
            req.frequency_penalty,
            req.presence_penalty,
            req.ignore_eos,
            req.seed,
            req.stop_token_ids,
        ),
        stop: req.stop,
        priority: RequestPriority::default(),
        delta_tx,
    };

    if let Err(e) = state.handle.submit(incoming) {
        error!("Scheduler unavailable or full: {e}");
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
                Event::default()
                    .data(serde_json::to_string(&chunk).expect("StreamChunk serialization")),
            )];

            if include_usage
                && is_terminal
                && let Some(usage) = usage
            {
                let usage_chunk =
                    StreamUsageChunk::from_usage(&request_id, created, &model_id, usage);
                events.push(Ok(Event::default().data(
                    serde_json::to_string(&usage_chunk).expect("StreamUsageChunk serialization"),
                )));
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

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    if req.messages.is_empty() {
        warn!("Rejecting empty messages request");
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens_or_default();
    let do_stream = req.stream_or_default();
    let include_usage = req.include_usage_or_default();
    let model_id = state.handle.model_id().to_string();

    // Convert messages → ChatML prompt.
    let prompt = crate::chat::messages_to_prompt(&req.messages, &req.tools);

    info!(
        "chat/completions: messages={}, prompt_len={}, max_tokens={}, stream={}",
        req.messages.len(),
        prompt.len(),
        max_tokens,
        do_stream,
    );

    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();

    let incoming = IncomingRequest {
        prompt,
        max_tokens,
        sampling: sampling_params_from_request(
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.repetition_penalty,
            req.frequency_penalty,
            req.presence_penalty,
            req.ignore_eos,
            req.seed,
            req.stop_token_ids,
        ),
        stop: req.stop,
        priority: RequestPriority::default(),
        delta_tx,
    };

    if let Err(e) = state.handle.submit(incoming) {
        error!("Scheduler unavailable or full: {e}");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    if do_stream {
        let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = now_secs();

        // First chunk carries the role field.
        let role_event = {
            let chunk = ChatStreamChunk::role_chunk(&request_id, created, &model_id);
            Ok::<_, Infallible>(
                Event::default()
                    .data(serde_json::to_string(&chunk).expect("ChatStreamChunk serialization")),
            )
        };

        let req_id = request_id;
        let mid = model_id.clone();
        let content_stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
            let usage = delta.usage;
            let is_terminal = delta.finish_reason.is_some();

            let chunk = ChatStreamChunk::content_chunk(&req_id, created, &mid, delta);
            let mut events = vec![Ok::<_, Infallible>(
                Event::default()
                    .data(serde_json::to_string(&chunk).expect("ChatStreamChunk serialization")),
            )];

            if include_usage
                && is_terminal
                && let Some(u) = usage
            {
                let usage_chunk = ChatStreamUsageChunk::from_usage(&req_id, created, &mid, u);
                events.push(Ok(Event::default().data(
                    serde_json::to_string(&usage_chunk)
                        .expect("ChatStreamUsageChunk serialization"),
                )));
            }

            stream::iter(events)
        });

        let done_stream =
            stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

        let full_stream = stream::once(async move { role_event })
            .chain(content_stream)
            .chain(done_stream);

        Ok(Sse::new(full_stream).into_response())
    } else {
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
            if let Some(r) = delta.finish_reason {
                finish_reason = r;
            }
            if let Some(u) = delta.usage {
                usage = u;
            }
        }

        info!(
            "chat/completions done: prompt_tokens={}, completion_tokens={}",
            usage.prompt_tokens, usage.completion_tokens
        );

        let response =
            ChatCompletionResponse::from_output(model_id, now_secs(), &text, finish_reason, usage);
        Ok(Json(response).into_response())
    }
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_prometheus();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_summary();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )],
        body,
    )
}

/// Build the Axum router with default (empty) metrics.
pub fn build_app(handle: SchedulerHandle) -> Router {
    build_app_with_metrics(handle, ServerMetrics::new(""))
}

/// Build the Axum router with a pre-configured `ServerMetrics` instance.
pub fn build_app_with_metrics(handle: SchedulerHandle, metrics: ServerMetrics) -> Router {
    let state = Arc::new(AppState { handle, metrics });

    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/metrics", get(metrics_handler))
        .route("/v1/stats", get(stats_handler))
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

    // -----------------------------------------------------------------------
    // /v1/chat/completions tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn chat_completion_basic() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Model id comes from the loaded model, not the request.
        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        // Content should contain something (mock returns "ok:<prompt>").
        assert!(
            payload["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "expected non-empty content, got: {payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_empty_messages() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_completion_streaming() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("chat.completion.chunk"),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"), "payload={payload}");
        // First chunk must carry role.
        assert!(
            payload.contains(r#""role":"assistant""#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("pegainfer_requests_total"),
            "payload={payload}"
        );
        assert!(
            payload.contains("pegainfer_requests_active"),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn stats_endpoint_returns_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();
        assert!(!payload.is_empty(), "stats body should not be empty");
    }

    #[tokio::test]
    async fn chat_completion_streaming_with_usage() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""prompt_tokens":1"#),
            "payload={payload}"
        );
    }
}
