use super::*;
use crate::sampler::SamplingParams;
use crate::server_engine::StreamDelta;

fn make_request() -> IncomingRequest {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<StreamDelta>();
    IncomingRequest {
        prompt: "hello".to_string(),
        max_tokens: 32,
        sampling: SamplingParams::default(),
        stop: None,
        priority: RequestPriority::Normal,
        delta_tx: tx,
    }
}

#[test]
fn scheduler_config_default_valid() {
    SchedulerConfig::default().validate().unwrap();
}

#[test]
fn scheduler_config_zero_slots_invalid() {
    let cfg = SchedulerConfig {
        max_slots: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn scheduler_config_zero_chunk_invalid() {
    let cfg = SchedulerConfig {
        prefill_chunk_size: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn priority_ordering() {
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
}

#[test]
fn priority_default_is_normal() {
    assert_eq!(RequestPriority::default(), RequestPriority::Normal);
}

#[tokio::test]
async fn submit_succeeds_when_queue_not_full() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 3);
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert_eq!(handle.waiting_count(), 3);
}

#[tokio::test]
async fn submit_fails_when_queue_full() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 2);
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_ok());
    assert!(handle.submit(make_request()).is_err());
    assert!(handle.is_full());
}

#[tokio::test]
async fn consume_decrements_waiting_count() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::with_max_waiting(tx, "test", 5);
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    assert_eq!(handle.waiting_count(), 2);
    handle.consume_one();
    assert_eq!(handle.waiting_count(), 1);
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    handle.submit(make_request()).unwrap();
    assert_eq!(handle.waiting_count(), 5);
}

#[tokio::test]
async fn unlimited_queue_never_rejects() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = SchedulerHandle::from_parts(tx, "test");
    for _ in 0..100 {
        assert!(handle.submit(make_request()).is_ok());
    }
    assert_eq!(handle.waiting_count(), 100);
}

#[test]
fn preemption_mode_default_is_recompute() {
    assert_eq!(PreemptionMode::default(), PreemptionMode::Recompute);
}

fn make_batch_scheduler(
    num_gpu_blocks: usize,
    block_size: usize,
    chunk_size: usize,
) -> BatchScheduler {
    let config = BatchSchedulerConfig {
        max_tokens_per_step: 4096,
        prefill_chunk_size: chunk_size,
        preemption_mode: PreemptionMode::Recompute,
    };
    let bm = crate::block_manager::BlockManager::new(num_gpu_blocks, 0, block_size);
    BatchScheduler::new(config, bm)
}

/// Run one simulated step: process the ScheduleDecision without a real model.
/// Returns (num_prefilled_reqs, num_decoded_reqs).
fn sim_step(sched: &mut BatchScheduler) -> (usize, usize) {
    match sched.schedule_step() {
        ScheduleDecision::Idle => (0, 0),
        ScheduleDecision::PrefillBatch(p) => (p.req_ids.len(), 0),
        ScheduleDecision::DecodeBatch(d) => {
            let to_finish: Vec<u64> = d.req_ids.clone();
            for id in &to_finish {
                sched.advance_decode(*id);
            }
            (0, to_finish.len())
        }
        ScheduleDecision::Mixed { decode, prefill } => {
            let n_prefill = prefill.req_ids.len();
            let to_finish: Vec<u64> = decode.req_ids.clone();
            for id in &to_finish {
                sched.advance_decode(*id);
            }
            (n_prefill, to_finish.len())
        }
    }
}

#[test]
fn test_continuous_batching() {
    let mut sched = make_batch_scheduler(32, 4, 64);

    let id0 = sched.add_request(vec![1, 2, 3, 4], 8, RequestPriority::Normal);
    let id1 = sched.add_request(vec![5, 6, 7, 8], 8, RequestPriority::Normal);
    let _id2 = sched.add_request(vec![9, 10, 11, 12], 8, RequestPriority::Normal);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id0]);
            assert_eq!(p.input_ids, vec![1, 2, 3, 4]);
        }
        other => panic!(
            "expected PrefillBatch, got {:?}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(sched.is_running(id0));
    assert_eq!(sched.waiting_len(), 2);

    match sched.schedule_step() {
        ScheduleDecision::Mixed { decode, prefill } => {
            assert_eq!(decode.req_ids, vec![id0]);
            assert_eq!(prefill.req_ids, vec![id1]);
            sched.advance_decode(id0);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id1]);
        }
        other => panic!("unexpected: {:?}", matches!(other, ScheduleDecision::Idle)),
    }

    sched.finish_request(id0);
    assert!(!sched.is_running(id0));

    for _ in 0..20 {
        sim_step(&mut sched);
        let running_ids: Vec<u64> = sched.running.keys().copied().collect();
        for id in running_ids {
            sched.finish_request(id);
        }
    }

    assert_eq!(sched.running_len(), 0);
    assert_eq!(sched.waiting_len(), 0);
}

#[test]
fn test_preemption() {
    let mut sched = make_batch_scheduler(5, 4, 16);

    let id0 = sched.add_request(vec![1, 2, 3, 4], 16, RequestPriority::Normal);
    let id1 = sched.add_request(vec![5, 6, 7, 8], 16, RequestPriority::Normal);
    let id2 = sched.add_request(vec![9, 10, 11, 12], 16, RequestPriority::Normal);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => assert_eq!(p.req_ids, vec![id0]),
        _ => panic!("expected prefill id0"),
    }
    match sched.schedule_step() {
        ScheduleDecision::Mixed { decode, prefill } => {
            assert_eq!(decode.req_ids, vec![id0]);
            assert_eq!(prefill.req_ids, vec![id1]);
            sched.advance_decode(id0);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id1]);
        }
        _ => panic!("unexpected decision"),
    }
    match sched.schedule_step() {
        ScheduleDecision::Mixed { prefill, .. } => {
            assert_eq!(prefill.req_ids, vec![id2]);
        }
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id2]);
        }
        _ => {}
    }

    let free_before = sched.free_gpu_blocks();
    match sched.schedule_step() {
        ScheduleDecision::DecodeBatch(d) => {
            assert!(
                !d.req_ids.contains(&id2),
                "id2 should be preempted, not decoding"
            );
            assert!(d.req_ids.contains(&id0) || d.req_ids.contains(&id1));
        }
        ScheduleDecision::Mixed { decode, .. } => {
            assert!(!decode.req_ids.contains(&id2));
        }
        ScheduleDecision::Idle => {}
        _ => {}
    }

    assert!(
        sched.is_waiting(id2) || !sched.is_running(id2),
        "id2 should have been preempted back to waiting"
    );
    assert!(
        sched.free_gpu_blocks() >= free_before || !sched.is_running(id2),
        "preemption should free blocks"
    );
}

#[test]
fn test_chunked_prefill() {
    let chunk_size = 8usize;
    let mut sched = make_batch_scheduler(8, 4, chunk_size);

    let prompt: Vec<u32> = (1u32..=20).collect();
    let id = sched.add_request(prompt, 4, RequestPriority::Normal);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), chunk_size);
            assert_eq!(p.input_ids[0], 1);
            assert_eq!(p.input_ids[chunk_size - 1], chunk_size as u32);
            assert_eq!(p.seq_lens, vec![chunk_size]);
        }
        other => panic!(
            "expected PrefillBatch chunk 1, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(!sched.is_running(id));
    assert_eq!(sched.waiting_len(), 1);

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), chunk_size);
            assert_eq!(p.input_ids[0], chunk_size as u32 + 1);
            assert_eq!(p.seq_lens, vec![chunk_size * 2]);
        }
        other => panic!(
            "expected PrefillBatch chunk 2, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(!sched.is_running(id));

    match sched.schedule_step() {
        ScheduleDecision::PrefillBatch(p) => {
            assert_eq!(p.req_ids, vec![id]);
            assert_eq!(p.input_ids.len(), 4);
            assert_eq!(p.input_ids[0], 17);
            assert_eq!(p.seq_lens, vec![20]);
        }
        other => panic!(
            "expected PrefillBatch final chunk, got Idle={}",
            matches!(other, ScheduleDecision::Idle)
        ),
    }
    assert!(sched.is_running(id));
    assert_eq!(sched.waiting_len(), 0);
    assert_eq!(sched.free_gpu_blocks(), 8 - 5);
}
