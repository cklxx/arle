use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use log::{error, info};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::model::{GenerationState, ModelForward};
use crate::paged_kv::PagedKVPool;
use crate::server_engine::{FinishReason, StreamDelta, Usage};
use crate::tokenizer::Tokenizer;

use super::{IncomingRequest, SchedulerHandle};

mod core;
mod decode;
mod execution;
mod prefill;
mod request;
mod runtime;

pub use core::Scheduler;
pub(super) use request::{ActiveRequest, Phase, StopCheckResult, check_stop_sequences};

pub(super) const PREFILL_CHUNK_SIZE: usize = 512;
/// Smaller chunk when decode is active, to avoid blocking decode latency.
pub(super) const PREFILL_CHUNK_SIZE_WITH_DECODE: usize = 64;
/// Interval (in completed requests) at which stats are logged.
pub(super) const STATS_LOG_INTERVAL: u64 = 10;
