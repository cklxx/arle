//! Dynamo distributed runtime integration.
//!
//! When compiled with `--features dynamo`, the agent-infer binary can register
//! itself with Dynamo's distributed runtime for service discovery and
//! KV-aware routing.
//!
//! ## Required environment
//!
//! The Dynamo runtime reads its configuration from environment variables
//! and/or config files. At minimum you need:
//!
//! - `NATS_SERVER` or the equivalent config — NATS connection for the message bus
//! - `ETCD_ENDPOINTS` — etcd cluster for service registration
//!
//! See the dynamo-runtime documentation for the full set of configuration knobs.

use std::sync::Arc;

use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    stream,
};
use tokio::sync::mpsc;

use infer::sampler::SamplingParams;
use infer::scheduler::{IncomingRequest, SchedulerHandle};

const NAMESPACE: &str = "dynamo";
const COMPONENT: &str = "agent-infer";
const ENDPOINT: &str = "generate";

/// Request handler that wraps agent-infer's generate capability and exposes
/// it as a Dynamo endpoint. Forwards requests to the infer scheduler.
struct AgentEndpoint {
    handle: SchedulerHandle,
}

impl AgentEndpoint {
    fn new(handle: SchedulerHandle) -> Arc<Self> {
        Arc::new(Self { handle })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for AgentEndpoint {
    async fn generate(
        &self,
        input: SingleIn<String>,
    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
        let (prompt, ctx) = input.into_parts();

        // Create a channel to receive streaming deltas from the scheduler.
        let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();

        let req = IncomingRequest {
            prompt,
            max_tokens: 256,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: None,
            delta_tx,
        };

        if !self.handle.submit(req) {
            anyhow::bail!("Scheduler is shut down");
        }

        // Collect all deltas into a Vec of Annotated<String>.
        let mut results = Vec::new();
        while let Some(delta) = delta_rx.recv().await {
            if !delta.text_delta.is_empty() {
                results.push(Annotated::from_data(delta.text_delta));
            }
            // Stop collecting once we get a finish signal.
            if delta.finish_reason.is_some() {
                break;
            }
        }

        let response_stream = stream::iter(results);
        Ok(ResponseStream::new(
            Box::pin(response_stream),
            ctx.context(),
        ))
    }
}

/// Register the agent-infer endpoint with the Dynamo distributed runtime.
///
/// This function blocks until the runtime shuts down (e.g. via signal).
async fn register(runtime: Runtime, handle: SchedulerHandle) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let ingress = Ingress::for_engine(AgentEndpoint::new(handle))?;

    let component = distributed.namespace(NAMESPACE)?.component(COMPONENT)?;
    component
        .endpoint(ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}

/// Entry point called from main when `--dynamo` is passed.
///
/// Creates a `Worker`, initialises the Dynamo runtime, registers the agent
/// as a discoverable endpoint, and blocks until shutdown.
pub fn run_dynamo_worker(handle: SchedulerHandle) -> anyhow::Result<()> {
    dynamo_runtime::logging::init();

    let worker = Worker::from_settings()?;
    worker.execute(move |runtime| register(runtime, handle))
}
