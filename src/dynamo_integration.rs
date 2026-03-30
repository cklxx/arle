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

const NAMESPACE: &str = "dynamo";
const COMPONENT: &str = "agent-infer";
const ENDPOINT: &str = "generate";

/// Request handler that wraps agent-infer's generate capability and exposes
/// it as a Dynamo endpoint.
///
/// For now this is a simple echo/passthrough handler that proves registration
/// works. The full integration would forward requests to the local
/// `ServerEngine` instance and stream back token-by-token responses.
struct AgentEndpoint;

impl AgentEndpoint {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for AgentEndpoint {
    async fn generate(
        &self,
        input: SingleIn<String>,
    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        // Placeholder: echo the request back as a single-token stream.
        // A full integration would run the agent loop here and stream tokens.
        let response = vec![Annotated::from_data(format!(
            "[agent-infer] received: {}",
            data
        ))];

        let stream = stream::iter(response);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

/// Register the agent-infer endpoint with the Dynamo distributed runtime.
///
/// This function blocks until the runtime shuts down (e.g. via signal).
async fn register(runtime: Runtime) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

    let ingress = Ingress::for_engine(AgentEndpoint::new())?;

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
pub fn run_dynamo_worker() -> anyhow::Result<()> {
    dynamo_runtime::logging::init();

    let worker = Worker::from_settings()?;
    worker.execute(register)
}
