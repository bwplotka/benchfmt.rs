// Copyright 2026 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Integration tests for [`benchfmt::ProcessAlloc`].

use std::hint::{black_box, spin_loop};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use benchfmt::{BenchConfig, BenchTime, ProcessAlloc, Suite};

#[global_allocator]
static ALLOC: ProcessAlloc = ProcessAlloc::new();

const SHUTDOWN: usize = usize::MAX;

struct WorkerPool {
    req: Arc<AtomicUsize>,
    resp: Arc<AtomicUsize>,
    handle: Option<thread::JoinHandle<()>>,
}

impl WorkerPool {
    fn spawn(mut work_fn: impl FnMut(usize) -> usize + Send + 'static) -> Self {
        let req = Arc::new(AtomicUsize::new(0));
        let resp = Arc::new(AtomicUsize::new(0));
        let worker_req = Arc::clone(&req);
        let worker_resp = Arc::clone(&resp);

        // Use lock-free atomic handoff rather than blocking channels so OS thread-parker lazy
        // initialization never allocates on the heap mid-benchmark.
        let handle = thread::spawn(move || {
            loop {
                let size = worker_req.load(Ordering::Acquire);
                if size == 0 {
                    spin_loop();
                    continue;
                }
                worker_req.store(0, Ordering::Relaxed);
                if size == SHUTDOWN {
                    break;
                }
                let out = work_fn(size);
                worker_resp.store(out, Ordering::Release);
            }
        });

        let pool = Self {
            req,
            resp,
            handle: Some(handle),
        };
        // Wait until the spawned thread is running its loop before starting any benchmark epoch.
        assert_eq!(pool.run_once(64), 64);
        pool
    }

    fn run_once(&self, size: usize) -> usize {
        self.resp.store(0, Ordering::Relaxed);
        self.req.store(size, Ordering::Release);
        loop {
            let done = self.resp.load(Ordering::Acquire);
            if done != 0 {
                return done;
            }
            spin_loop();
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.req.store(SHUTDOWN, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[test]
fn process_alloc_counts_worker_thread_allocations_across_suite() {
    let pool = WorkerPool::spawn(|size| {
        let buf = black_box(vec![7u8; size]);
        buf.len()
    });

    let cfg = BenchConfig {
        benchtime: BenchTime::Iterations(25),
        count: 2,
        cpu_procs: 2,
        benchmem: true,
        pkg_override: Some("multithread/pkg".to_owned()),
        ..BenchConfig::default()
    };

    let mut out = Vec::new();
    Suite::with_config("ignored", cfg)
        .with_alloc(&ALLOC)
        .bench("WorkerAlloc/size=4kb", |b| {
            // Setup allocation outside `keep_running` must be excluded from `B/op`.
            let unmeasured = black_box(vec![1u8; 16 * 1024]);
            black_box(unmeasured.len());
            while b.keep_running() {
                assert_eq!(pool.run_once(4096), 4096);
                b.report_metric(4096.0, "worker-B/op");
            }
        })
        .run_to(&mut out)
        .unwrap();

    drop(pool);

    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("pkg: multithread/pkg\n"), "{text}");
    assert!(
        text.contains("BenchmarkWorkerAlloc/size=4kb-2\t      25\t"),
        "{text}"
    );
    assert!(
        text.contains("    4096 B/op\t       1 allocs/op\t      4096 worker-B/op"),
        "{text}"
    );
    assert!(text.ends_with("PASS\n"), "{text}");

    #[cfg(feature = "pprof")]
    {
        use std::fs;
        use std::io::Read;

        let dir = std::env::temp_dir().join(format!(
            "benchfmt_process_alloc_pprof_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mem_path = dir.join("mem.pb.gz");

        let pprof_pool = WorkerPool::spawn(|size| {
            let buf = black_box(worker_heap_alloc_helper(size));
            buf.len()
        });

        let pprof_cfg = BenchConfig {
            benchtime: BenchTime::Iterations(20),
            count: 1,
            cpu_procs: 2,
            benchmem: true,
            mem_profile: Some(mem_path.clone()),
            mem_profile_rate: 1,
            ..BenchConfig::default()
        };

        let mut pprof_out = Vec::new();
        Suite::with_config("multithread_pprof", pprof_cfg)
            .with_alloc(&ALLOC)
            .bench("WorkerMemProfile", |b| {
                while b.keep_running() {
                    assert_eq!(pprof_pool.run_once(2048), 2048);
                }
            })
            .run_to(&mut pprof_out)
            .unwrap();

        drop(pprof_pool);

        let pprof_text = String::from_utf8(pprof_out).unwrap();
        assert!(
            pprof_text.contains("    2048 B/op\t       1 allocs/op"),
            "internal heap sampling allocations leaked into ProcessAlloc counters:\n{pprof_text}"
        );

        let mem_gz = fs::read(&mem_path).unwrap();
        let mut mem_raw = Vec::new();
        flate2::read::GzDecoder::new(&mem_gz[..])
            .read_to_end(&mut mem_raw)
            .unwrap();
        let mem_lossy = String::from_utf8_lossy(&mem_raw);
        assert!(
            mem_lossy.contains("worker_heap_alloc_helper"),
            "expected worker-thread allocation helper in memprofile"
        );
        assert!(
            !mem_lossy.contains("ProcessAlloc")
                && !mem_lossy.contains("record_alloc")
                && !mem_lossy.contains("capture_heap_sample")
                && !mem_lossy.contains("alloc_impl"),
            "internal ProcessAlloc or allocator frames leaked into memprofile"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "pprof")]
#[inline(never)]
fn worker_heap_alloc_helper(size: usize) -> Vec<u8> {
    vec![42u8; size]
}
