# benchfmt.rs

👋 I'm coming from the Go ecosystem and I can't live without Go's [`benchstat`-- it's too amazing](https://www.bwplotka.dev/2024/go-microbenchmarks-benchstat/). This repo brings solid benchmarking practices and experience from [`golang.org/x/perf/benchfmt`](https://pkg.go.dev/golang.org/x/perf/benchfmt) to Rust!

See [`benchfmt.rs: Bringing Go's Benchmarking and benchstat Workflow to Rust`](https://www.bwplotka.dev/2026/rust-microbenchmarks-benchstat/) blog post for deeper explanation and fun Go vs Rust learnings.

### What this repo contains?

1. `Criterion`/`libtest` converter emitting Go's [Benchmark Data Format](https://go.googlesource.com/proposal/+/master/design/14313-benchmark-format.md) for direct analysis with [`benchstat`](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat).
1. Replacement `benchfmt::Suite`: zero-dependency Rust benchmark runner, with heap allocation counter and `pprof` profile support, mimicking `go test -bench` framework features and CLI surface.

### Motivation

Comparing Rust changes across commits or benchmarking Rust implementations side by side and against Go code hits three practical gaps:

1. `benchstat` expects Go's `benchfmt` line format with `ns/op`, decimal `MB/s`, `B/op`, `allocs/op`, independent `-count` epochs for Mann-Whitney U tests, and `/key=value` sub-benchmark dimensions.
2. Criterion measures wall-clock time and throughput, leaving out per-operation heap bytes and allocation counts.
3. Go's `testing.B` calibrates an iteration count `N` for a fixed `-benchtime` such as `2s` and repeats `-count` independent epochs, whereas Criterion defaults to 100 linear-ramp samples.

### When to switch to `benchfmt.rs` replacement `Suite` vs wrapping `Criterion`/`libtest` outputs?

- Use the `benchfmt criterion` CLI when a repository already has Criterion benchmarks and you want `benchstat` tables across commits or `/key=value` dimensions without touching benchmark code, accepting that Criterion omits heap allocations and that the converter approximates epochs by bucketing Criterion's sample array.
- Use `benchfmt libtest` for quick one-off `benchfmt` formatting of `cargo +nightly bench` output. Because `libtest` prints a single summary line per benchmark, one run does not give `benchstat` multiple epochs for p-values unless you concatenate repeated runs.
- Use standalone Criterion when you want built-in HTML and SVG violin plots, async executor adapters, or local baseline comparisons without installing Go's `benchstat`.
- **Use `Suite` and `CountingAlloc` in a `harness = false` benchmark binary when writing new microbenchmarks, catching `B/op` and `allocs/op` regressions, capturing scoped `-cpuprofile` and `-memprofile` pprof files, comparing Rust and Go implementations on the same machine, or keeping default `cargo bench` compile times near zero**.

| Aspect | `benchfmt::Suite` | `benchfmt criterion` converter | Standalone Criterion |
| --- | --- | --- | --- |
| External Rust crates | `0` by default on `std`, optional `pprof` feature | `0` in CLI, Criterion in `dev-dependencies` | ~40+ crates |
| Sampling model | `-count` independent calibrated epochs for fixed `-benchtime`, e.g. `6 x 2s` | 100 linear-ramp samples bucketed into `--epochs`, `6` by default | 100 linear-ramp or flat samples |
| Fit for `benchstat` Mann-Whitney U | Exact, each epoch is an independent steady-state run | Approximate, early buckets average shorter cold runs and later buckets average longer warm runs | N/A, uses Criterion's own t-test and bootstrap |
| Heap allocations | `B/op` and `allocs/op` via `CountingAlloc` or `ProcessAlloc` | Not recorded | Not recorded |
| Custom metrics | `b.report_metric(value, "unit")` (e.g. `wire-B/op`, `items/s`), plus built-in metric overrides | Not recorded | Single `Measurement` trait per group |
| Profiling | Optional `-cpuprofile` and `-memprofile` pprof flags plus `Profiler` trait | None | External `pprof` or custom `Profiler` adapter |
| Throughput unit | Decimal `MB/s` at `1e6` bytes/s, matching Go `testing.B` | Converts `Throughput::Bytes` to decimal `MB/s` | Decimal `MB/s` or binary `MiB/s` |
| Multi-run comparison | `benchstat -col /dim` across arbitrary files, commits, and Go vs. Rust | `benchstat -col /dim` via `full_id` | Pairwise `base` vs. `new` directory |

### Usage

#### CLI converter usage

```bash
cargo install --git https://github.com/bwplotka/benchfmt.rs

# Run Criterion benchmarks as usual, then emit 6-epoch benchfmt lines:
cargo bench
benchfmt criterion target/criterion --pkg expfmt --cpu 2 > rust.txt
benchstat old.txt rust.txt
```

#### `benchfmt::Suite` Runner usage

In `Cargo.toml`:

```toml
[dev-dependencies]
benchfmt = { git = "https://github.com/bwplotka/benchfmt.rs" }

[[bench]]
name = "decode"
harness = false
```

In `benches/decode.rs`:

```rust
use std::hint::black_box;
use benchfmt::{CountingAlloc, Suite};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc::new();

fn main() {
    let data = vec![b'a'; 64 * 1024];
    Suite::from_env("my_crate")
        .with_alloc(&ALLOC)
        .bench("Decode/parser=rust", |b| {
            b.set_bytes(data.len() as u64);
            b.report_allocs();
            while b.keep_running() {
                black_box(data.len());
            }
            b.report_metric(data.len() as f64, "wire-B/op");
        })
        .run();
}
```

For multi-threaded benchmarks where iterations dispatch work onto worker threads (such as a Tokio runtime or Rayon pool), replace `CountingAlloc` with `ProcessAlloc` so allocations across all threads are counted in `B/op` and `allocs/op`:

```rust
use benchfmt::{ProcessAlloc, Suite};

#[global_allocator]
static ALLOC: ProcessAlloc = ProcessAlloc::new();
```

Run with Go-compatible flags and compare with `benchstat`:

```bash
cargo bench --bench decode -- -bench '^BenchmarkDecode' -benchtime 2s -count 6 -cpu 2 -benchmem | tee rust.txt
benchstat -col /parser -ignore pkg go.txt rust.txt
```

##### CPU and heap pprof profiling

Enable the optional `pprof` feature and line-table debuginfo in `Cargo.toml` because Cargo defaults `[profile.bench]` to `debug = false`:

```toml
[dev-dependencies]
benchfmt = { git = "https://github.com/bwplotka/benchfmt.rs", features = ["pprof"] }

[profile.bench]
debug = 1
```

Pass `-cpuprofile` on Unix or `-memprofile` on any target, then inspect with `go tool pprof`:

```bash
cargo bench --bench decode -- -bench '^BenchmarkDecode' -benchtime 2s -cpuprofile cpu.pb.gz -memprofile mem.pb.gz
go tool pprof -http=:8080 cpu.pb.gz
go tool pprof -http=:8080 mem.pb.gz
```

`-memprofilerate` controls the mean bytes allocated per heap sample, defaulting to `524288` for 512 KiB just like Go's `runtime.MemProfileRate`. Passing `-memprofilerate 1` records every heap allocation. Custom profilers can also hook into [`Suite::with_profiler`](src/lib.rs) via the zero-dependency [`Profiler`](src/lib.rs) trait.

##### Scope compared to Go's `testing.B`

- `-bench` strips leading `^`, trailing `$`, and `Benchmark`, then matches names with `str::contains` or exact equality under `--exact`, rather than full regular expressions. `--skip` excludes matching substrings.
- `-cpu N` sets the `-N` suffix on `BenchmarkFoo-N` lines rather than sweeping a comma-separated thread list or tuning a runtime scheduler.
- `CountingAlloc` tracks `B/op` and `allocs/op` on the calling thread without atomic contention, while `ProcessAlloc` aggregates `B/op` and `allocs/op` across all threads for multi-threaded benchmarks; `-memprofile` samples heap allocations across active threads during `keep_running()` with either allocator.
- Benchmarks register flat names on `Suite::bench` such as `"Decode/case=1kb"` rather than nesting `b.Run` sub-closures, and `Bencher` exposes `reset_timer()` for one-time setup before the loop rather than per-iteration `StopTimer` / `StartTimer`.
- `Bencher::report_metric(value, unit)` mirrors Go's `b.ReportMetric`: custom units are sorted alphabetically, reporting the same unit overwrites the previous value, reporting `"ns/op"`, `"MB/s"`, `"B/op"`, or `"allocs/op"` overrides the built-in metric, and setting `"ns/op"` or `"MB/s"` to `0.0` suppresses it.

### License

Apache-2.0
