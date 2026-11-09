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

//! Go [Benchmark Data Format (`benchfmt`)](https://go.googlesource.com/proposal/+/master/design/14313-benchmark-format.md)
//! harness, heap allocation tracker, and Criterion/`libtest` converter for Rust.
//!
//! Designed for apples-to-apples performance regression testing (`benchstat old.txt new.txt`)
//! and cross-language Go vs. Rust benchmark comparisons on the same machine, reporting
//! `ns/op`, decimal `MB/s`, `B/op`, `allocs/op`, and custom metrics via
//! [`Bencher::report_metric`].
//!
//! # Example (`harness = false` benchmark binary)
//!
//! ```no_run
//! use std::hint::black_box;
//! use benchfmt::{CountingAlloc, Suite};
//!
//! #[global_allocator]
//! static ALLOC: CountingAlloc = CountingAlloc::new();
//!
//! fn main() {
//!     let payload = vec![b'x'; 4096];
//!     Suite::from_env("my_crate")
//!         .with_alloc(&ALLOC)
//!         .bench("ScanBytes/impl=rust", |b| {
//!             b.set_bytes(payload.len() as u64);
//!             b.report_allocs();
//!             while b.keep_running() {
//!                 black_box(payload.iter().filter(|&&c| c == b'x').count());
//!             }
//!         })
//!         .run();
//! }
//! ```

mod alloc;
mod convert;
mod json;
#[cfg(feature = "pprof")]
mod pprof_proto;
#[cfg(feature = "pprof")]
mod pprof_support;

use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

pub use alloc::{AllocSnapshot, AllocTracker, CountingAlloc, ProcessAlloc};
pub use convert::{ConvertOptions, convert_criterion_dir, convert_libtest_output};

const MAX_ITERATIONS: u64 = 1_000_000_000;

/// Default Poisson mean byte interval between heap profile samples (`512 KiB`), matching Go's
/// `runtime.MemProfileRate`.
pub const DEFAULT_MEM_PROFILE_RATE: usize = 512 * 1024;

/// Target benchmark duration or fixed iteration count (`-benchtime`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchTime {
    /// Calibrate iteration count until a single run takes at least this [`Duration`].
    Duration(Duration),
    /// Run an exact number of iterations (`-benchtime=1000x`).
    Iterations(u64),
}

impl Default for BenchTime {
    fn default() -> Self {
        Self::Duration(Duration::from_secs(1))
    }
}

impl BenchTime {
    /// Parses a Go `-benchtime` argument such as `"2s"`, `"500ms"`, `"100us"`, or `"1000x"`.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if let Some(iters) = trimmed.strip_suffix('x') {
            let n = iters
                .parse::<u64>()
                .map_err(|_| format!("invalid iteration benchtime {raw:?}"))?;
            if n == 0 || n > i64::MAX as u64 {
                return Err("iteration benchtime must be in 1..=i64::MAX".to_owned());
            }
            return Ok(Self::Iterations(n));
        }
        let dur = parse_duration(trimmed)?;
        Ok(Self::Duration(dur))
    }
}

fn parse_duration(raw: &str) -> Result<Duration, String> {
    if let Some(ms) = raw.strip_suffix("ms") {
        let v: f64 = ms
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        return secs_to_nonzero_duration(v / 1_000.0, raw);
    }
    if let Some(us) = raw.strip_suffix("us").or_else(|| raw.strip_suffix("µs")) {
        let v: f64 = us
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        return secs_to_nonzero_duration(v / 1_000_000.0, raw);
    }
    if let Some(ns) = raw.strip_suffix("ns") {
        let v: u64 = ns
            .parse()
            .map_err(|_| format!("invalid duration {raw:?}"))?;
        if v == 0 {
            return Err("duration benchtime must be > 0".to_owned());
        }
        return Ok(Duration::from_nanos(v));
    }
    if let Some(m) = raw.strip_suffix('m') {
        let v: f64 = m.parse().map_err(|_| format!("invalid duration {raw:?}"))?;
        return secs_to_nonzero_duration(v * 60.0, raw);
    }
    if let Some(s) = raw.strip_suffix('s') {
        let v: f64 = s.parse().map_err(|_| format!("invalid duration {raw:?}"))?;
        return secs_to_nonzero_duration(v, raw);
    }
    Err(format!(
        "unsupported duration unit in {raw:?} (expected ns, us, ms, s, m, or Nx)"
    ))
}

fn secs_to_nonzero_duration(secs: f64, raw: &str) -> Result<Duration, String> {
    if !secs.is_finite() || secs <= 0.0 {
        return Err(format!("invalid duration {raw:?} (must be finite and > 0)"));
    }
    let dur = Duration::try_from_secs_f64(secs).map_err(|_| format!("invalid duration {raw:?}"))?;
    if dur.is_zero() {
        return Err(format!("invalid duration {raw:?} (must be > 0)"));
    }
    Ok(dur)
}

/// Top-of-file Go `benchfmt` configuration metadata (`goos`, `goarch`, `pkg`, `cpu`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchHeader {
    /// Operating system name matching `GOOS` (`linux`, `darwin`, `windows`, ...).
    pub goos: String,
    /// Architecture name matching `GOARCH` (`amd64`, `arm64`, ...).
    pub goarch: String,
    /// Package or crate identifier (`pkg: ...`).
    pub pkg: String,
    /// CPU model string (`cpu: ...`), when detectable.
    pub cpu: Option<String>,
    /// Extra `key: value` configuration lines (for example `impl: rust`).
    pub extra: Vec<(String, String)>,
}

impl BenchHeader {
    /// Detects host `goos`, `goarch`, and CPU model for `pkg`.
    #[must_use]
    pub fn detect(pkg: impl Into<String>) -> Self {
        let goos = match std::env::consts::OS {
            "macos" => "darwin",
            other => other,
        }
        .to_owned();
        let goarch = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            "x86" => "386",
            other => other,
        }
        .to_owned();
        Self {
            goos,
            goarch,
            pkg: pkg.into(),
            cpu: detect_cpu_model(),
            extra: Vec::new(),
        }
    }

    fn validate(&self) -> io::Result<()> {
        validate_header_kv("goos", &self.goos)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        validate_header_kv("goarch", &self.goarch)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        validate_header_kv("pkg", &self.pkg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        if let Some(cpu) = &self.cpu {
            let trimmed_cpu = cpu.trim();
            if !trimmed_cpu.is_empty() {
                validate_header_kv("cpu", trimmed_cpu)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            }
        }
        for (k, v) in &self.extra {
            validate_header_kv(k, v).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        }
        Ok(())
    }

    /// Writes the configuration block in `benchfmt` format.
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if any header key violates Go's `benchfmt` key
    /// grammar (lowercase start, no uppercase/whitespace/colons) or any value contains newlines or
    /// is empty.
    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        self.validate()?;

        writeln!(out, "goos: {}", self.goos.trim())?;
        writeln!(out, "goarch: {}", self.goarch.trim())?;
        writeln!(out, "pkg: {}", self.pkg.trim())?;
        if let Some(cpu) = &self.cpu {
            let trimmed_cpu = cpu.trim();
            if !trimmed_cpu.is_empty() {
                writeln!(out, "cpu: {trimmed_cpu}")?;
            }
        }
        for (k, v) in &self.extra {
            writeln!(out, "{k}: {}", v.trim())?;
        }
        Ok(())
    }
}

/// Validates a Go `benchfmt` `key: value` header pair (`golang.org/x/perf/benchfmt`).
#[doc(hidden)]
pub fn validate_header_kv(key: &str, value: &str) -> Result<(), String> {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return Err("benchfmt header key must not be empty".to_owned());
    };
    if !first.is_lowercase() {
        return Err(format!(
            "benchfmt header key {key:?} must start with a lowercase letter"
        ));
    }
    if chars.any(|c| c.is_uppercase() || c.is_whitespace() || c.is_control() || c == ':') {
        return Err(format!(
            "benchfmt header key {key:?} must not contain uppercase, whitespace, control characters, or ':'"
        ));
    }
    if value.chars().any(|c| matches!(c, '\r' | '\n')) {
        return Err(format!(
            "benchfmt header value for {key:?} must not contain newlines"
        ));
    }
    if value.trim().is_empty() {
        return Err(format!(
            "benchfmt header value for {key:?} must not be empty"
        ));
    }
    Ok(())
}

fn detect_cpu_model() -> Option<String> {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for key in ["model name", "Model", "Hardware", "uarch"] {
            for line in cpuinfo.lines() {
                if let Some((lhs, rhs)) = line.split_once(':')
                    && lhs.trim() == key
                {
                    let trimmed = rhs.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_owned());
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        && out.status.success()
        && let Ok(raw) = String::from_utf8(out.stdout)
    {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }
    None
}

/// Validates a `(unit, value)` pair reported via [`Bencher::report_metric`] or [`BenchRecord`].
fn validate_metric(unit: &str, value: f64) -> Result<(), String> {
    if unit.is_empty() {
        return Err("benchmark metric unit must not be empty".to_owned());
    }
    if unit.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!(
            "benchmark metric unit {unit:?} must not contain whitespace or control characters"
        ));
    }
    if !value.is_finite() {
        return Err(format!(
            "benchmark metric value for {unit:?} must be finite, got {value}"
        ));
    }
    if matches!(unit, "ns/op" | "MB/s" | "B/op" | "allocs/op") && value < 0.0 {
        return Err(format!(
            "built-in benchmark metric {unit:?} must be >= 0, got {value}"
        ));
    }
    if matches!(unit, "B/op" | "allocs/op") && value >= u64::MAX as f64 {
        return Err(format!(
            "built-in benchmark metric {unit:?} exceeds u64::MAX, got {value}"
        ));
    }
    Ok(())
}

/// Formats a `(value, unit)` pair using Go `testing/benchmark.go`'s `prettyPrint` widths and
/// 4-significant-digit thresholds.
fn write_pretty_metric(mut out: impl Write, x: f64, unit: &str) -> io::Result<()> {
    let x = if x == 0.0 { 0.0 } else { x };
    let y = x.abs();
    if y == 0.0 || y >= 999.95 {
        write!(out, "{x:10.0} {unit}")
    } else if y >= 99.995 {
        write!(out, "{x:12.1} {unit}")
    } else if y >= 9.9995 {
        write!(out, "{x:13.2} {unit}")
    } else if y >= 0.99995 {
        write!(out, "{x:14.3} {unit}")
    } else if y >= 0.099995 {
        write!(out, "{x:15.4} {unit}")
    } else if y >= 0.0099995 {
        write!(out, "{x:16.5} {unit}")
    } else if y >= 0.00099995 {
        write!(out, "{x:17.6} {unit}")
    } else {
        write!(out, "{x:18.7} {unit}")
    }
}

/// Single formatted `Benchmark...` result line matching Go's `testing.BenchmarkResult`.
#[derive(Clone, Debug)]
pub struct BenchRecord<'a> {
    /// Benchmark name without the leading `Benchmark` prefix (added automatically if absent).
    pub name: &'a str,
    /// Optional `-N` CPU/GOMAXPROCS suffix (omitted when `None` or `Some(1)`).
    pub cpu_procs: Option<usize>,
    /// Number of iterations executed in this epoch (`b.N`).
    pub iterations: u64,
    /// Mean nanoseconds per operation.
    pub ns_per_op: f64,
    /// Optional decimal throughput in megabytes per second (`1e6` bytes/sec).
    pub mb_per_sec: Option<f64>,
    /// Optional heap bytes allocated per operation (`B/op`).
    pub alloc_bytes_per_op: Option<u64>,
    /// Optional heap allocations per operation (`allocs/op`).
    pub allocs_per_op: Option<u64>,
    /// Additional user-reported `(unit, value)` metrics from [`Bencher::report_metric`].
    pub extra_metrics: Vec<(String, f64)>,
}

impl BenchRecord<'_> {
    /// Writes this benchmark measurement line in Go `benchfmt` syntax.
    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        if !self.ns_per_op.is_finite() || self.ns_per_op < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("ns_per_op must be finite and >= 0, got {}", self.ns_per_op),
            ));
        }
        if self.iterations > i64::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "iterations must be <= i64::MAX for Go benchfmt, got {}",
                    self.iterations
                ),
            ));
        }

        let mut ns_override: Option<f64> = None;
        let mut mbps_override: Option<f64> = None;
        let mut bytes_override: Option<u64> = None;
        let mut allocs_override: Option<u64> = None;
        let mut custom: Vec<(&str, f64)> = Vec::new();

        for (unit, value) in &self.extra_metrics {
            validate_metric(unit, *value)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            match unit.as_str() {
                "ns/op" => ns_override = Some(*value),
                "MB/s" => mbps_override = Some(*value),
                "B/op" => bytes_override = Some(*value as u64),
                "allocs/op" => allocs_override = Some(*value as u64),
                other => {
                    if let Some((_, existing)) = custom.iter_mut().find(|(u, _)| *u == other) {
                        *existing = *value;
                    } else {
                        custom.push((other, *value));
                    }
                }
            }
        }
        custom.sort_unstable_by_key(|(a, _)| *a);

        // Match Go's `BenchmarkResult.String`: setting `"ns/op"` to `0.0` via `report_metric`
        // suppresses the built-in `ns/op` column, but a valid `benchfmt` record must still emit at
        // least one `<value> <unit>` pair.
        let emit_ns = match ns_override {
            Some(0.0) => None,
            Some(v) => Some(v),
            None => Some(self.ns_per_op),
        };
        let emit_mbps = mbps_override
            .or(self.mb_per_sec)
            .filter(|&v| v.is_finite() && v > 0.0);
        let emit_bytes = bytes_override.or(self.alloc_bytes_per_op);
        let emit_allocs = allocs_override.or(self.allocs_per_op);

        if emit_ns.is_none()
            && emit_mbps.is_none()
            && emit_bytes.is_none()
            && emit_allocs.is_none()
            && custom.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "benchmark record must emit at least one metric when ns/op is suppressed",
            ));
        }

        let sanitized = sanitize_bench_name(self.name);
        let full_name = match self.cpu_procs {
            Some(procs) if procs > 1 => format!("Benchmark{sanitized}-{procs}"),
            _ => format!("Benchmark{sanitized}"),
        };

        write!(out, "{full_name}\t{:8}", self.iterations.max(1))?;
        if let Some(ns) = emit_ns {
            write!(out, "\t")?;
            write_pretty_metric(&mut out, ns, "ns/op")?;
        }
        if let Some(mbps) = emit_mbps {
            write!(out, "\t{mbps:8.2} MB/s")?;
        }
        if let Some(bytes) = emit_bytes {
            write!(out, "\t{bytes:8} B/op")?;
        }
        if let Some(allocs) = emit_allocs {
            write!(out, "\t{allocs:8} allocs/op")?;
        }
        for (unit, value) in custom {
            write!(out, "\t")?;
            write_pretty_metric(&mut out, value, unit)?;
        }
        writeln!(out)
    }
}

/// Strips leading `"Benchmark"` or `"benchmark"` prefixes when they act as Go's `Benchmark*`
/// boundary (empty remainder or followed by a non-lowercase character), preserving words like
/// `"Benchmarking"` while keeping [`sanitize_bench_name`] idempotent.
fn strip_benchmark_prefix(mut s: &str) -> &str {
    while let Some(rest) = s
        .strip_prefix("Benchmark")
        .or_else(|| s.strip_prefix("benchmark"))
    {
        if rest.chars().next().is_none_or(|ch| !ch.is_lowercase()) {
            s = rest;
        } else {
            break;
        }
    }
    s
}

/// Normalizes a Rust or Criterion benchmark identifier into a whitespace-free Go `Benchmark*`
/// suffix while preserving `/key=value` sub-benchmark dimensions for `benchstat`.
#[must_use]
pub fn sanitize_bench_name(raw: &str) -> String {
    let trimmed = strip_benchmark_prefix(raw.trim());
    let mut out = String::with_capacity(trimmed.len() + 1);
    let mut chars = trimmed.chars();
    if let Some(first) = chars.next() {
        if first.is_whitespace() || first.is_control() {
            out.push('_');
        } else {
            out.extend(first.to_uppercase());
        }
    }
    for ch in chars {
        if ch.is_whitespace() || ch.is_control() {
            out.push('_');
        } else {
            out.push(ch);
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

const SUITE_USAGE: &str = "\
Usage: <bench-binary> [OPTIONS] [FILTER]

Go-compatible options:
  -bench, -test.bench PATTERN             Run benchmarks matching PATTERN (`.` runs all).
  -benchtime, -test.benchtime DURATION    Target duration (`2s`, `500ms`) or count (`1000x`).
  -count, -test.count N                   Repeat each benchmark N epochs (default: 1).
  -cpu, -test.cpu N                       Set `-N` CPU suffix on benchmark names.
  -benchmem, -test.benchmem               Always report `B/op` and `allocs/op`.
  -cpuprofile, -test.cpuprofile FILE      Write CPU pprof profile (`pprof` feature, Unix).
  -memprofile, -test.memprofile FILE      Write heap pprof profile (`pprof` feature).
  -memprofilerate, -test.memprofilerate N Mean bytes per heap sample (default: 524288).
  -run, -test.run PATTERN                 Accepted for `go test` compatibility (ignored).

Cargo / libtest options:
  --pkg PKG                               Override `pkg:` metadata header line.
  --list                                  List benchmarks without running them.
  --exact                                 Require exact benchmark name match.
  --skip PATTERN                          Skip benchmarks matching PATTERN.
  --ignored                               Skip non-ignored benchmarks (libtest compatibility).
  --test                                  Run 1 iteration per benchmark as a smoke test.
  -h, --help                              Print this help text.
";

/// Runtime options parsed from CLI arguments for [`Suite`].
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Substring (or `--exact`) filter applied to benchmark names (`-bench` / `--filter`).
    pub filter: Option<String>,
    /// Patterns of benchmark names to skip (`--skip`).
    pub skip: Vec<String>,
    /// Require `filter` to match the full benchmark name rather than a substring (`--exact`).
    pub exact: bool,
    /// Target duration or iteration count per epoch (`-benchtime`).
    pub benchtime: BenchTime,
    /// Number of independent epochs to execute (`-count`).
    pub count: usize,
    /// Reported CPU suffix (`-cpu`), defaulting to available parallelism.
    pub cpu_procs: usize,
    /// Force allocation reporting on all benchmarks (`-benchmem`).
    pub benchmem: bool,
    /// Override `pkg:` header line (`--pkg`).
    pub pkg_override: Option<String>,
    /// List benchmark names without running them (`--list`).
    pub list_only: bool,
    /// Run a single-iteration smoke test (`--test` compatibility executes one `1x` epoch per
    /// benchmark).
    pub test_mode: bool,
    /// Skip all non-ignored benchmarks (`--ignored` `libtest` compatibility).
    pub run_ignored_only: bool,
    /// Print CLI usage text and exit without running benchmarks (`-h` / `--help`).
    pub show_help: bool,
    /// Write a pprof CPU profile to this path (`-cpuprofile`).
    pub cpu_profile: Option<PathBuf>,
    /// Write a pprof heap allocation profile to this path (`-memprofile`).
    pub mem_profile: Option<PathBuf>,
    /// Mean bytes allocated per heap sample (`-memprofilerate`; `1` samples every allocation, `0`
    /// disables sampling).
    pub mem_profile_rate: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        let cpu_procs = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self {
            filter: None,
            skip: Vec::new(),
            exact: false,
            benchtime: BenchTime::default(),
            count: 1,
            cpu_procs,
            benchmem: false,
            pkg_override: None,
            list_only: false,
            test_mode: false,
            run_ignored_only: false,
            show_help: false,
            cpu_profile: None,
            mem_profile: None,
            mem_profile_rate: DEFAULT_MEM_PROFILE_RATE,
        }
    }
}

impl BenchConfig {
    /// Parses Go-style (`-bench`, `-benchtime`, `-count`, `-cpu`, `-benchmem`, `-cpuprofile`,
    /// `-memprofile`, `-memprofilerate`) and Cargo-style (`--bench`, `--list`, positional filter)
    /// CLI flags.
    pub fn from_args<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cfg = Self::default();
        let mut iter = args.into_iter();
        while let Some(raw) = iter.next() {
            let arg = raw.as_ref();
            if matches!(arg, "-h" | "--help" | "-test.h") {
                cfg.show_help = true;
                continue;
            }
            if matches!(
                arg,
                "--bench"
                    | "--nocapture"
                    | "--no-capture"
                    | "--show-output"
                    | "--quiet"
                    | "-q"
                    | "--include-ignored"
                    | "--report-time"
                    | "--ensure-time"
                    | "-v"
                    | "-test.v"
                    | "-test.paniconexit0"
            ) || arg.starts_with("-test.v=")
                || arg.starts_with("-test.paniconexit0=")
            {
                continue;
            }
            if arg == "--test" {
                cfg.test_mode = true;
                continue;
            }
            if arg == "--list" {
                cfg.list_only = true;
                continue;
            }
            if arg == "--exact" {
                cfg.exact = true;
                continue;
            }
            if arg == "--ignored" {
                cfg.run_ignored_only = true;
                continue;
            }
            if matches!(arg, "-benchmem" | "--benchmem" | "-test.benchmem") {
                cfg.benchmem = true;
                continue;
            }
            if let Some(val) = arg
                .strip_prefix("-benchmem=")
                .or_else(|| arg.strip_prefix("--benchmem="))
                .or_else(|| arg.strip_prefix("-test.benchmem="))
            {
                cfg.benchmem = match val {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => return Err(format!("invalid -benchmem value {val:?}")),
                };
                continue;
            }
            if split_flag(
                arg,
                &[
                    "-run",
                    "--run",
                    "-test.run",
                    "--color",
                    "--format",
                    "--test-threads",
                    "--logfile",
                    "-timeout",
                    "-test.timeout",
                    "-test.gocoverdir",
                ],
                &mut iter,
            )?
            .is_some()
            {
                continue;
            }
            if let Some(val) = split_flag(arg, &["--skip"], &mut iter)? {
                if !val.is_empty() {
                    cfg.skip.push(val);
                }
                continue;
            }
            if let Some(val) = split_flag(arg, &["-bench", "--filter", "-test.bench"], &mut iter)? {
                if matches!(val.as_str(), "." | "^" | "$" | ".*" | "^.*$") {
                    cfg.filter = None;
                } else if val.is_empty() || val == "^$" {
                    cfg.filter = Some("^$".to_owned());
                } else {
                    let unanchored = val.trim_start_matches('^').trim_end_matches('$');
                    cfg.filter = Some(strip_benchmark_prefix(unanchored).to_owned());
                }
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &["-benchtime", "--benchtime", "-test.benchtime"],
                &mut iter,
            )? {
                cfg.benchtime = BenchTime::parse(&val)?;
                continue;
            }
            if let Some(val) = split_flag(arg, &["-count", "--count", "-test.count"], &mut iter)? {
                cfg.count = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -count {val:?}"))?
                    .max(1);
                continue;
            }
            if let Some(val) = split_flag(arg, &["-cpu", "--cpu", "-test.cpu"], &mut iter)? {
                cfg.cpu_procs = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -cpu {val:?}"))?
                    .max(1);
                continue;
            }
            if let Some(val) = split_flag(arg, &["-pkg", "--pkg"], &mut iter)? {
                validate_header_kv("pkg", &val)?;
                cfg.pkg_override = Some(val);
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &["-cpuprofile", "--cpuprofile", "-test.cpuprofile"],
                &mut iter,
            )? {
                cfg.cpu_profile = Some(PathBuf::from(val));
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &["-memprofile", "--memprofile", "-test.memprofile"],
                &mut iter,
            )? {
                cfg.mem_profile = Some(PathBuf::from(val));
                continue;
            }
            if let Some(val) = split_flag(
                arg,
                &[
                    "-memprofilerate",
                    "--memprofilerate",
                    "-test.memprofilerate",
                ],
                &mut iter,
            )? {
                cfg.mem_profile_rate = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid -memprofilerate {val:?}"))?;
                continue;
            }
            if arg.starts_with('-') {
                return Err(format!("unknown flag {arg:?}"));
            }
            if cfg.filter.is_some() {
                return Err(format!("unexpected extra positional argument {arg:?}"));
            }
            cfg.filter = Some(strip_benchmark_prefix(arg).to_owned());
        }
        Ok(cfg)
    }
}

fn matches_bench_pattern(pattern: &str, exact: bool, raw_name: &str, sanitized_name: &str) -> bool {
    if pattern == "^$" {
        return false;
    }
    let full_name = format!("Benchmark{sanitized_name}");
    let stripped = strip_benchmark_prefix(pattern);
    if exact {
        pattern == sanitized_name
            || pattern == raw_name
            || pattern == full_name
            || stripped == sanitized_name
            || stripped == raw_name
    } else {
        sanitized_name.contains(pattern)
            || raw_name.contains(pattern)
            || full_name.contains(pattern)
            || (!stripped.is_empty()
                && (sanitized_name.contains(stripped) || raw_name.contains(stripped)))
    }
}

fn split_flag<I, S>(arg: &str, names: &[&str], iter: &mut I) -> Result<Option<String>, String>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    for &name in names {
        if arg == name {
            let val = iter
                .next()
                .ok_or_else(|| format!("missing argument for {name}"))?;
            return Ok(Some(val.as_ref().to_owned()));
        }
        if let Some(rest) = arg.strip_prefix(name).and_then(|s| s.strip_prefix('=')) {
            return Ok(Some(rest.to_owned()));
        }
    }
    Ok(None)
}

/// Scoped profiler notified around the timed portion of each benchmark run.
///
/// [`Bencher`] invokes [`Profiler::start_bench`] before recording its allocation baseline and
/// start timestamp, and invokes [`Profiler::stop_bench`] after recording elapsed time and
/// allocation deltas so profiler work does not skew `ns/op`, `B/op`, or `allocs/op`.
pub trait Profiler {
    /// Starts profiling for a timed `keep_running` window of `bench_name`.
    fn start_bench(&mut self, bench_name: &str) -> io::Result<()>;

    /// Stops profiling for the current timed window of `bench_name`.
    fn stop_bench(&mut self, bench_name: &str) -> io::Result<()>;

    /// Flushes any accumulated profile output once all benchmarks finish.
    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Per-epoch state passed to a benchmark closure, mirroring Go's `*testing.B`.
pub struct Bencher<'a> {
    bench_name: &'a str,
    target_n: u64,
    remaining: u64,
    started: bool,
    fallback_start: Instant,
    fallback_start_alloc: AllocSnapshot,
    start_instant: Option<Instant>,
    elapsed: Duration,
    bytes_per_op: Option<u64>,
    report_allocs: bool,
    alloc_tracker: Option<&'a dyn AllocTracker>,
    start_alloc: AllocSnapshot,
    alloc_delta: AllocSnapshot,
    extra_metrics: Vec<(String, f64)>,
    profiler: Option<&'a mut dyn Profiler>,
    profiler_running: bool,
    profiler_err: Option<io::Error>,
}

impl<'a> Bencher<'a> {
    fn new(
        bench_name: &'a str,
        target_n: u64,
        default_benchmem: bool,
        alloc_tracker: Option<&'a dyn AllocTracker>,
        profiler: Option<&'a mut dyn Profiler>,
    ) -> Self {
        let fallback_start_alloc =
            alloc_tracker.map_or_else(AllocSnapshot::default, |t| t.snapshot());
        Self {
            bench_name,
            target_n,
            remaining: target_n,
            started: false,
            fallback_start: Instant::now(),
            fallback_start_alloc,
            start_instant: None,
            elapsed: Duration::ZERO,
            bytes_per_op: None,
            report_allocs: default_benchmem,
            alloc_tracker,
            start_alloc: AllocSnapshot::default(),
            alloc_delta: AllocSnapshot::default(),
            extra_metrics: Vec::new(),
            profiler,
            profiler_running: false,
            profiler_err: None,
        }
    }

    /// Returns the calibrated target iteration count (`b.N` in Go) for the current epoch.
    ///
    /// Prefer [`Bencher::keep_running`] or [`Bencher::iter`] so setup before the loop stays
    /// excluded and scoped [`Profiler`] windows are bracketed automatically. After
    /// [`Bencher::keep_running`] finishes, `b.iterations()` can be used as the divisor when
    /// computing per-operation [`Bencher::report_metric`] averages.
    #[must_use]
    pub const fn iterations(&self) -> u64 {
        self.target_n
    }

    /// Sets the number of bytes processed in a single operation so `MB/s` is reported.
    pub fn set_bytes(&mut self, bytes: u64) {
        self.bytes_per_op = Some(bytes);
    }

    /// Enables `B/op` and `allocs/op` reporting when the [`Suite`] has an [`AllocTracker`].
    pub fn report_allocs(&mut self) {
        self.report_allocs = true;
    }

    /// Adds a user-reported `(value, unit)` metric to the benchmark results, matching Go's
    /// `testing.B.ReportMetric`.
    ///
    /// If the metric is accumulated across the loop, divide by `b.iterations() as f64` after
    /// [`Bencher::keep_running`] finishes, and by convention end per-operation units in `"/op"`
    /// (for example `"wire-B/op"` or `"cache-misses/op"`). Reporting the same `unit` multiple
    /// times within a run overrides the previous value. Reporting a built-in unit (`"ns/op"`,
    /// `"MB/s"`, `"B/op"`, or `"allocs/op"`) overrides that metric, and setting `"ns/op"` or
    /// `"MB/s"` to `0.0` suppresses it.
    ///
    /// An empty `unit`, a `unit` containing whitespace or control characters, or a non-finite
    /// `value` causes [`Suite::run_to`] to return [`io::ErrorKind::InvalidInput`].
    pub fn report_metric(&mut self, value: f64, unit: &str) {
        if let Err(msg) = validate_metric(unit, value) {
            if self.start_instant.is_some() {
                self.remaining = 0;
            }
            if self.profiler_err.is_none() {
                self.profiler_err = Some(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("benchmark {:?}: {msg}", self.bench_name),
                ));
            }
            return;
        }
        alloc::without_alloc_tracking(|| {
            if let Some((_, existing)) = self.extra_metrics.iter_mut().find(|(u, _)| u == unit) {
                *existing = value;
            } else {
                self.extra_metrics.push((unit.to_owned(), value));
            }
        });
    }

    /// Resets the elapsed timer and allocation baseline before the benchmark loop starts,
    /// excluding any prior one-time setup work.
    ///
    /// Calling `reset_timer` inside an active [`Bencher::keep_running`] loop is unsupported
    /// (matching Go 1.24's `b.Loop()`) and causes [`Suite::run_to`] to return
    /// [`io::ErrorKind::InvalidInput`].
    pub fn reset_timer(&mut self) {
        if self.started {
            if self.start_instant.is_some() {
                // Abort the loop immediately so per-iteration `reset_timer()` calls never hang or
                // mismeasure `target_n + 1` iterations.
                self.remaining = 0;
                if self.profiler_err.is_none() {
                    self.profiler_err = Some(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "benchmark {:?} called reset_timer() inside keep_running(); move one-time setup before keep_running()",
                            self.bench_name
                        ),
                    ));
                }
            }
            return;
        }
        if let Some(tracker) = self.alloc_tracker {
            let snap = tracker.snapshot();
            self.start_alloc = snap;
            self.fallback_start_alloc = snap;
        }
        self.elapsed = Duration::ZERO;
        self.fallback_start = Instant::now();
    }

    /// Go 1.24 `b.Loop()` equivalent: starts timing and allocation tracking on the first call,
    /// runs `N` iterations, and stops timing + snapshots allocations as soon as the loop finishes.
    #[inline]
    pub fn keep_running(&mut self) -> bool {
        if !self.started {
            self.started = true;
            // NOTE: Start the profiler before capturing allocation and wall-clock baselines so any
            // profiler initialization cost stays outside the measured benchmark window.
            if let Some(prof) = self.profiler.as_deref_mut() {
                if let Err(err) = prof.start_bench(self.bench_name) {
                    self.profiler_err = Some(err);
                    return false;
                }
                self.profiler_running = true;
            }
            if let Some(tracker) = self.alloc_tracker {
                self.start_alloc = tracker.snapshot();
            }
            self.start_instant = Some(Instant::now());
        }
        if self.remaining == 0 {
            if let Some(started_at) = self.start_instant.take() {
                self.elapsed += started_at.elapsed();
                if let Some(tracker) = self.alloc_tracker {
                    self.alloc_delta = self.start_alloc.delta_to(tracker.snapshot());
                }
            }
            // NOTE: Stop the profiler only after wall-clock and allocation deltas are recorded.
            if self.profiler_running {
                self.profiler_running = false;
                if let Some(prof) = self.profiler.as_deref_mut()
                    && let Err(err) = prof.stop_bench(self.bench_name)
                    && self.profiler_err.is_none()
                {
                    self.profiler_err = Some(err);
                }
            }
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Executes `f` for the calibrated number of iterations, preventing return-value dead-code
    /// elimination with [`std::hint::black_box`].
    pub fn iter<R>(&mut self, mut f: impl FnMut() -> R) {
        while self.keep_running() {
            black_box(f());
        }
    }
}

impl Drop for Bencher<'_> {
    fn drop(&mut self) {
        if self.profiler_running {
            self.profiler_running = false;
            if let Some(prof) = self.profiler.as_deref_mut() {
                let _ = prof.stop_bench(self.bench_name);
            }
        }
    }
}

type BenchFn<'a> = Box<dyn FnMut(&mut Bencher<'_>) + 'a>;

struct RegisteredBench<'a> {
    raw_name: String,
    sanitized_name: String,
    func: BenchFn<'a>,
}

/// Collection of benchmarks configured to emit Go `benchfmt` output.
pub struct Suite<'a> {
    header: BenchHeader,
    config: BenchConfig,
    alloc_tracker: Option<&'a dyn AllocTracker>,
    profiler: Option<&'a mut dyn Profiler>,
    benches: Vec<RegisteredBench<'a>>,
}

impl<'a> Suite<'a> {
    /// Creates a [`Suite`] configured from `std::env::args().skip(1)`.
    #[must_use]
    pub fn from_env(default_pkg: impl Into<String>) -> Self {
        let cfg = BenchConfig::from_args(std::env::args().skip(1)).unwrap_or_else(|err| {
            eprintln!("benchfmt argument error: {err}\n\n{SUITE_USAGE}");
            std::process::exit(2);
        });
        Self::with_config(default_pkg, cfg)
    }

    /// Creates a [`Suite`] with an explicit [`BenchConfig`].
    #[must_use]
    pub fn with_config(default_pkg: impl Into<String>, config: BenchConfig) -> Self {
        let pkg = config
            .pkg_override
            .clone()
            .unwrap_or_else(|| default_pkg.into());
        Self {
            header: BenchHeader::detect(pkg),
            config,
            alloc_tracker: None,
            profiler: None,
            benches: Vec::new(),
        }
    }

    /// Replaces the [`BenchHeader`] emitted at the start of the suite output.
    #[must_use]
    pub fn with_header(mut self, mut header: BenchHeader) -> Self {
        if let Some(pkg) = &self.config.pkg_override {
            header.pkg.clone_from(pkg);
        }
        self.header = header;
        self
    }

    /// Attaches a global [`AllocTracker`] (such as [`CountingAlloc`] or [`ProcessAlloc`]) so
    /// `B/op` and `allocs/op` can be measured.
    #[must_use]
    pub fn with_alloc(mut self, tracker: &'a dyn AllocTracker) -> Self {
        self.alloc_tracker = Some(tracker);
        self
    }

    /// Attaches a custom [`Profiler`] scoped to the timed portion of each benchmark run.
    #[must_use]
    pub fn with_profiler(mut self, profiler: &'a mut dyn Profiler) -> Self {
        self.profiler = Some(profiler);
        self
    }

    /// Adds a custom `key: value` metadata header line (for example `("impl", "rust")`).
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.header.extra.push((key.into(), value.into()));
        self
    }

    /// Registers a benchmark function under `name`.
    #[must_use]
    pub fn bench(mut self, name: impl Into<String>, f: impl FnMut(&mut Bencher<'_>) + 'a) -> Self {
        let raw_name = name.into();
        let sanitized_name = sanitize_bench_name(&raw_name);
        self.benches.push(RegisteredBench {
            raw_name,
            sanitized_name,
            func: Box::new(f),
        });
        self
    }

    /// Runs all matching benchmarks and writes `benchfmt` output to stdout.
    pub fn run(&mut self) {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        if let Err(err) = self.run_to(&mut handle) {
            eprintln!("benchfmt error: {err}");
            std::process::exit(1);
        }
    }

    /// Runs all matching benchmarks and writes `benchfmt` output to `out`.
    pub fn run_to(&mut self, mut out: impl Write) -> io::Result<()> {
        if self.config.show_help {
            write!(out, "{SUITE_USAGE}")?;
            return out.flush();
        }
        if self.config.list_only {
            for b in &self.benches {
                if !self.should_run_bench(&b.raw_name, &b.sanitized_name) {
                    continue;
                }
                writeln!(out, "Benchmark{}: benchmark", b.sanitized_name)?;
            }
            return Ok(());
        }

        #[cfg(not(feature = "pprof"))]
        if self.config.cpu_profile.is_some() || self.config.mem_profile.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "-cpuprofile and -memprofile require enabling the `pprof` Cargo feature on `benchfmt`",
            ));
        }

        self.header.validate()?;

        #[cfg(feature = "pprof")]
        let mut builtin_pprof = pprof_support::PprofSession::new(&self.config)?;

        self.header.write(&mut out)?;
        out.flush()?;

        let effective_benchtime = if self.config.test_mode {
            BenchTime::Iterations(1)
        } else {
            self.config.benchtime
        };
        let effective_count = if self.config.test_mode {
            1
        } else {
            self.config.count.max(1)
        };

        for idx in 0..self.benches.len() {
            if !self.should_run_bench(
                &self.benches[idx].raw_name,
                &self.benches[idx].sanitized_name,
            ) {
                continue;
            }
            let bench = &mut self.benches[idx];
            let name = &bench.sanitized_name;
            let func = &mut bench.func;
            for _ in 0..effective_count {
                #[cfg(feature = "pprof")]
                let rec = {
                    let has_any = builtin_pprof.is_some() || self.profiler.is_some();
                    let mut combined = pprof_support::CombinedProfiler {
                        builtin: builtin_pprof.as_mut(),
                        custom: self
                            .profiler
                            .as_mut()
                            .map(|p| &mut **p as &mut dyn Profiler),
                    };
                    let prof_arg: Option<&mut dyn Profiler> =
                        if has_any { Some(&mut combined) } else { None };
                    run_single_epoch(
                        name,
                        effective_benchtime,
                        self.config.cpu_procs,
                        self.config.benchmem,
                        self.alloc_tracker,
                        prof_arg,
                        func,
                    )?
                };
                #[cfg(not(feature = "pprof"))]
                let rec = run_single_epoch(
                    name,
                    effective_benchtime,
                    self.config.cpu_procs,
                    self.config.benchmem,
                    self.alloc_tracker,
                    self.profiler
                        .as_mut()
                        .map(|p| &mut **p as &mut dyn Profiler),
                    func,
                )?;
                rec.write(&mut out)?;
                out.flush()?;
            }
        }

        #[cfg(feature = "pprof")]
        if builtin_pprof.is_some() || self.profiler.is_some() {
            let mut combined = pprof_support::CombinedProfiler {
                builtin: builtin_pprof.as_mut(),
                custom: self
                    .profiler
                    .as_mut()
                    .map(|p| &mut **p as &mut dyn Profiler),
            };
            combined.finish()?;
        }
        #[cfg(not(feature = "pprof"))]
        if let Some(prof) = self.profiler.as_deref_mut() {
            prof.finish()?;
        }

        writeln!(out, "PASS")?;
        out.flush()
    }

    fn should_run_bench(&self, raw_name: &str, sanitized_name: &str) -> bool {
        if self.config.run_ignored_only {
            return false;
        }
        if let Some(filter) = &self.config.filter
            && !matches_bench_pattern(filter, self.config.exact, raw_name, sanitized_name)
        {
            return false;
        }
        if self
            .config
            .skip
            .iter()
            .any(|pat| matches_bench_pattern(pat, false, raw_name, sanitized_name))
        {
            return false;
        }
        true
    }
}

fn run_single_epoch<'a>(
    name: &'a str,
    benchtime: BenchTime,
    cpu_procs: usize,
    default_benchmem: bool,
    alloc_tracker: Option<&dyn AllocTracker>,
    mut profiler: Option<&mut dyn Profiler>,
    func: &mut BenchFn<'_>,
) -> io::Result<BenchRecord<'a>> {
    let mut n = match benchtime {
        BenchTime::Iterations(fixed) => fixed.max(1),
        BenchTime::Duration(_) => 1,
    };

    loop {
        let (elapsed, bytes_per_op, report_allocs, alloc_delta, extra_metrics) = {
            let mut bencher = Bencher::new(
                name,
                n,
                default_benchmem,
                alloc_tracker,
                profiler.as_mut().map(|p| &mut **p as &mut dyn Profiler),
            );
            func(&mut bencher);
            // Finalize timing if the benchmark body exited without exhausting `keep_running`.
            if let Some(started_at) = bencher.start_instant.take() {
                bencher.elapsed += started_at.elapsed();
                if let Some(tracker) = bencher.alloc_tracker {
                    bencher.alloc_delta = bencher.start_alloc.delta_to(tracker.snapshot());
                }
            } else if !bencher.started {
                // Support legacy `for _ in 0..b.iterations()` closures that do not call
                // `keep_running()`.
                bencher.elapsed = bencher.fallback_start.elapsed();
                if let Some(tracker) = bencher.alloc_tracker {
                    bencher.alloc_delta = bencher.fallback_start_alloc.delta_to(tracker.snapshot());
                }
            }
            if bencher.profiler_running {
                bencher.profiler_running = false;
                if let Some(prof) = bencher.profiler.as_deref_mut()
                    && let Err(err) = prof.stop_bench(name)
                    && bencher.profiler_err.is_none()
                {
                    bencher.profiler_err = Some(err);
                }
            }
            if let Some(err) = bencher.profiler_err.take() {
                return Err(err);
            }
            if bencher.started && bencher.remaining != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "benchmark {name:?} exited keep_running() early with {} of {n} iterations remaining",
                        bencher.remaining
                    ),
                ));
            }
            (
                bencher.elapsed,
                bencher.bytes_per_op,
                bencher.report_allocs,
                bencher.alloc_delta,
                std::mem::take(&mut bencher.extra_metrics),
            )
        };

        let done = match benchtime {
            BenchTime::Iterations(_) => true,
            BenchTime::Duration(target) => elapsed >= target || n >= MAX_ITERATIONS,
        };

        if done {
            let elapsed_ns = elapsed.as_nanos() as f64;
            let ns_per_op = elapsed_ns / (n as f64);
            let mb_per_sec = bytes_per_op.and_then(|bytes| {
                if elapsed_ns > 0.0 {
                    Some((bytes as f64 * n as f64 * 1_000.0) / elapsed_ns)
                } else {
                    None
                }
            });
            let (alloc_bytes_per_op, allocs_per_op) = if report_allocs && alloc_tracker.is_some() {
                (Some(alloc_delta.bytes / n), Some(alloc_delta.allocs / n))
            } else {
                (None, None)
            };
            return Ok(BenchRecord {
                name,
                cpu_procs: Some(cpu_procs),
                iterations: n,
                ns_per_op,
                mb_per_sec,
                alloc_bytes_per_op,
                allocs_per_op,
                extra_metrics,
            });
        }

        let target_ns = match benchtime {
            BenchTime::Duration(d) => d.as_nanos(),
            BenchTime::Iterations(_) => unreachable!(),
        };
        let elapsed_ns = elapsed.as_nanos();
        let next = if let Some(predicted) = target_ns
            .saturating_mul(u128::from(n))
            .checked_div(elapsed_ns)
        {
            let predicted = u64::try_from(predicted).unwrap_or(u64::MAX);
            // Grow by 20% above the linear prediction, capped at 100x per step, matching Go.
            let padded = predicted.saturating_add(predicted / 5);
            padded.min(n.saturating_mul(100)).max(n + 1)
        } else {
            n.saturating_mul(100)
        };
        n = next.min(MAX_ITERATIONS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[global_allocator]
    static TEST_ALLOC: CountingAlloc = CountingAlloc::new();

    #[test]
    fn counting_alloc_and_suite_emit_valid_benchfmt() {
        let cfg = BenchConfig {
            filter: Some("AllocVec".to_owned()),
            benchtime: BenchTime::Iterations(50),
            count: 2,
            cpu_procs: 2,
            benchmem: false,
            pkg_override: Some("example/pkg".to_owned()),
            list_only: false,
            ..BenchConfig::default()
        };
        let mut out = Vec::new();
        Suite::with_config("ignored", cfg)
            .with_alloc(&TEST_ALLOC)
            .bench("AllocVec/case=1kb", |b| {
                b.set_bytes(1024);
                b.report_allocs();
                while b.keep_running() {
                    let v = vec![1u8; 1024];
                    black_box(v);
                }
            })
            .bench("SkippedBench", |b| {
                b.iter(|| 1 + 1);
            })
            .run_to(&mut out)
            .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("pkg: example/pkg\n"), "{text}");
        assert!(
            text.contains("BenchmarkAllocVec/case=1kb-2\t      50\t"),
            "{text}"
        );
        assert!(text.contains("1024 B/op\t       1 allocs/op"), "{text}");
        assert!(!text.contains("SkippedBench"), "{text}");
        assert!(text.ends_with("PASS\n"), "{text}");
    }

    #[test]
    fn converts_criterion_directory_to_benchfmt() {
        let tmp = std::env::temp_dir().join(format!("benchfmt_test_{}", std::process::id()));
        let new_dir = tmp.join("ParsePromText/rust/new");
        fs::create_dir_all(&new_dir).unwrap();
        fs::write(
            new_dir.join("benchmark.json"),
            r#"{"group_id":"ParsePromText","function_id":"parser=rust","value_str":null,"throughput":{"Bytes":1000},"full_id":"ParsePromText/parser=rust","directory_name":"ParsePromText/rust","title":"ParsePromText/parser=rust"}"#,
        )
        .unwrap();
        fs::write(
            new_dir.join("sample.json"),
            r#"{"sampling_mode":"Linear","iters":[10.0,20.0,30.0,40.0],"times":[1000.0,2000.0,3000.0,4000.0]}"#,
        )
        .unwrap();

        let opts = ConvertOptions {
            header: BenchHeader {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                pkg: "expfmt".to_owned(),
                cpu: None,
                extra: Vec::new(),
            },
            cpu_procs: Some(2),
            criterion_epochs: 2,
            criterion_baseline: "new".to_owned(),
        };
        let mut out = Vec::new();
        let count = convert_criterion_dir(&tmp, &opts, &mut out).unwrap();
        fs::remove_dir_all(&tmp).unwrap();

        assert_eq!(count, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains(
                "BenchmarkParsePromText/parser=rust-2\t      30\t       100.0 ns/op\t10000.00 MB/s"
            ),
            "{text}"
        );
    }

    #[test]
    fn converts_libtest_output_to_benchfmt() {
        let sample = "running 1 test\ntest parse::promtext ... bench:   1,250.50 ns/iter (+/- 14.00) = 152.10 MB/s\n";
        let mut out = Vec::new();
        let count = convert_libtest_output(sample, &ConvertOptions::default(), &mut out).unwrap();
        assert_eq!(count, 1);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("BenchmarkParse::promtext"), "{text}");
        assert!(text.contains("1250 ns/op"), "{text}");
        assert!(text.contains("152.10 MB/s"), "{text}");
    }

    #[test]
    fn parses_profile_cli_flags() {
        let cfg = BenchConfig::from_args([
            "-cpuprofile",
            "cpu.pb.gz",
            "-test.memprofile=mem.pb.gz",
            "--memprofilerate",
            "4096",
        ])
        .unwrap();
        assert_eq!(
            cfg.cpu_profile.as_deref(),
            Some(std::path::Path::new("cpu.pb.gz"))
        );
        assert_eq!(
            cfg.mem_profile.as_deref(),
            Some(std::path::Path::new("mem.pb.gz"))
        );
        assert_eq!(cfg.mem_profile_rate, 4096);
    }

    #[test]
    fn parses_benchtime_units_and_rejects_invalid_values() {
        assert_eq!(
            BenchTime::parse("250ns").unwrap(),
            BenchTime::Duration(Duration::from_nanos(250))
        );
        assert_eq!(
            BenchTime::parse("100us").unwrap(),
            BenchTime::Duration(Duration::from_micros(100))
        );
        assert_eq!(
            BenchTime::parse("100µs").unwrap(),
            BenchTime::Duration(Duration::from_micros(100))
        );
        assert_eq!(
            BenchTime::parse("50ms").unwrap(),
            BenchTime::Duration(Duration::from_millis(50))
        );
        assert_eq!(
            BenchTime::parse("2s").unwrap(),
            BenchTime::Duration(Duration::from_secs(2))
        );
        assert_eq!(
            BenchTime::parse("0.5m").unwrap(),
            BenchTime::Duration(Duration::from_secs(30))
        );
        assert_eq!(
            BenchTime::parse("1000x").unwrap(),
            BenchTime::Iterations(1_000)
        );

        for bad in [
            "0x", "0s", "0ns", "-1s", "-5ms", "NaNs", "infs", "1e300s", "10h", "",
        ] {
            assert!(
                BenchTime::parse(bad).is_err(),
                "expected error for invalid benchtime {bad:?}"
            );
        }
    }

    #[test]
    fn parses_go_and_cargo_cli_flags_cleanly() {
        // `-test.run` / `-run` consume their pattern and `-bench .` clears any earlier filter.
        let cfg = BenchConfig::from_args([
            "-run",
            "^$",
            "-test.run",
            "TestNothing",
            "-bench",
            "^BenchmarkDecode$",
            "-bench",
            ".",
            "--color",
            "never",
            "--format=terse",
            "--nocapture",
            "--exact",
            "--skip",
            "Slow",
        ])
        .unwrap();
        assert_eq!(cfg.filter, None);
        assert!(cfg.exact);
        assert_eq!(cfg.skip, vec!["Slow"]);

        // Boolean `libtest` flags (`--report-time`, `--ensure-time`, `--no-capture`) do not
        // swallow a trailing positional filter, and Go `-test.benchmem=true/false` is supported.
        let libtest_bool = BenchConfig::from_args([
            "--report-time",
            "--ensure-time",
            "--no-capture",
            "-test.benchmem=true",
            "Decode",
        ])
        .unwrap();
        assert_eq!(libtest_bool.filter.as_deref(), Some("Decode"));
        assert!(libtest_bool.benchmem);
        assert!(
            !BenchConfig::from_args(["-benchmem=false"])
                .unwrap()
                .benchmem
        );
        assert!(BenchConfig::from_args(["-test.benchmem=maybe"]).is_err());

        // `-bench '^$'` matches nothing rather than everything.
        let empty_pat = BenchConfig::from_args(["-bench", "^$"]).unwrap();
        assert_eq!(empty_pat.filter.as_deref(), Some("^$"));

        // Unknown flags, invalid `--pkg`, and extra positional args fail fast.
        assert!(BenchConfig::from_args(["-benchtimee", "2s"]).is_err());
        assert!(BenchConfig::from_args(["--pkg", ""]).is_err());
        assert!(BenchConfig::from_args(["--pkg", "bad\npkg"]).is_err());
        assert!(BenchConfig::from_args(["FilterOne", "FilterTwo"]).is_err());
    }

    #[test]
    fn sanitizes_names_and_validates_headers_and_precision_tiers() {
        assert_eq!(sanitize_bench_name("Benchmarking/sub"), "Benchmarking/sub");
        assert_eq!(sanitize_bench_name("benchmarking/sub"), "Benchmarking/sub");
        assert_eq!(sanitize_bench_name("Benchmarks"), "Benchmarks");
        assert_eq!(sanitize_bench_name("BenchmarkBenchmarkFoo"), "Foo");
        assert_eq!(sanitize_bench_name("benchmark_decode"), "_decode");
        for sample in [
            "Benchmarking/sub",
            "benchmarking/sub",
            "BenchmarkBenchmarkFoo",
            "benchmark_decode",
            "Benchmark",
            "benchmark",
        ] {
            let once = sanitize_bench_name(sample);
            assert_eq!(sanitize_bench_name(&once), once, "non-idempotent: {sample}");
        }
        assert_eq!(sanitize_bench_name("decode\tline\ninj"), "Decode_line_inj");
        assert_eq!(sanitize_bench_name("   "), "_");
        assert_eq!(sanitize_bench_name("Benchmark"), "_");

        // Header validation rejects uppercase keys, colons, spaces, and newline values.
        for (k, v) in [
            ("IMPL", "rust"),
            ("bad key", "v"),
            ("bad:key", "v"),
            ("", "v"),
            ("impl", ""),
            ("impl", "line1\nline2"),
        ] {
            let hdr = BenchHeader {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                pkg: "pkg".to_owned(),
                cpu: None,
                extra: vec![(k.to_owned(), v.to_owned())],
            };
            assert!(
                hdr.write(Vec::new()).is_err(),
                "expected invalid header for ({k:?}, {v:?})"
            );
        }

        // Go `prettyPrint` 4-sig-fig formatting across magnitudes.
        let cases = [
            (1250.4, "      1250 ns/op"),
            (100.4, "       100.4 ns/op"),
            (12.346, "        12.35 ns/op"),
            (1.2346, "         1.235 ns/op"),
            (0.12346, "         0.1235 ns/op"),
            (0.0012346, "         0.001235 ns/op"),
        ];
        for (ns_per_op, expected) in cases {
            let mut buf = Vec::new();
            BenchRecord {
                name: "Fmt",
                cpu_procs: Some(1),
                iterations: 10,
                ns_per_op,
                mb_per_sec: None,
                alloc_bytes_per_op: None,
                allocs_per_op: None,
                extra_metrics: Vec::new(),
            }
            .write(&mut buf)
            .unwrap();
            let line = String::from_utf8(buf).unwrap();
            assert!(
                line.contains(expected),
                "got {line:?}, expected {expected:?}"
            );
        }

        assert!(
            BenchRecord {
                name: "Bad",
                cpu_procs: None,
                iterations: 1,
                ns_per_op: f64::NAN,
                mb_per_sec: None,
                alloc_bytes_per_op: None,
                allocs_per_op: None,
                extra_metrics: Vec::new(),
            }
            .write(Vec::new())
            .is_err()
        );
        assert!(
            BenchRecord {
                name: "OverflowIters",
                cpu_procs: None,
                iterations: (i64::MAX as u64) + 1,
                ns_per_op: 1.0,
                mb_per_sec: None,
                alloc_bytes_per_op: None,
                allocs_per_op: None,
                extra_metrics: Vec::new(),
            }
            .write(Vec::new())
            .is_err()
        );
    }

    #[test]
    fn suite_duration_calibration_reset_timer_and_bencher_contracts() {
        // Duration calibration + lowercase filter matching + `reset_timer` + `with_tag`.
        let cfg = BenchConfig {
            filter: Some("decode".to_owned()),
            skip: vec!["ignored_case".to_owned()],
            benchtime: BenchTime::Duration(Duration::from_millis(5)),
            count: 1,
            cpu_procs: 1,
            benchmem: true,
            ..BenchConfig::default()
        };
        let mut out = Vec::new();
        Suite::with_config("pkg", cfg)
            .with_alloc(&TEST_ALLOC)
            .with_tag("impl", "rust")
            .bench("decode/case=fast", |b| {
                let _setup = vec![1u8; 4096];
                b.reset_timer();
                while b.keep_running() {
                    black_box(vec![2u8; 128]);
                }
                // Extra post-loop `keep_running()` and `reset_timer()` calls must not overwrite
                // `elapsed` or `alloc_delta`.
                let _after = vec![3u8; 8192];
                assert!(!b.keep_running());
                b.reset_timer();
                black_box(_after);
            })
            .bench("decode/case=ignored_case", |b| b.iter(|| 1))
            .run_to(&mut out)
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("impl: rust\n"), "{text}");
        assert!(text.contains("BenchmarkDecode/case=fast\t"), "{text}");
        assert!(text.contains("     128 B/op\t       1 allocs/op"), "{text}");
        assert!(!text.contains("ignored_case"), "{text}");

        // Legacy `for _ in 0..b.iterations()` loop and `Iterations(0)` clamping.
        let mut legacy_out = Vec::new();
        Suite::with_config(
            "pkg",
            BenchConfig {
                benchtime: BenchTime::Iterations(0),
                benchmem: true,
                cpu_procs: 1,
                ..BenchConfig::default()
            },
        )
        .with_alloc(&TEST_ALLOC)
        .bench("LegacyLoop", |b| {
            b.reset_timer();
            for _ in 0..b.iterations() {
                black_box(vec![5u8; 64]);
            }
        })
        .run_to(&mut legacy_out)
        .unwrap();
        let legacy_text = String::from_utf8(legacy_out).unwrap();
        assert!(
            legacy_text.contains("      64 B/op\t       1 allocs/op"),
            "{legacy_text}"
        );

        // Early `break` inside `keep_running()` returns an InvalidInput error.
        let err = Suite::with_config(
            "pkg",
            BenchConfig {
                benchtime: BenchTime::Iterations(10),
                ..BenchConfig::default()
            },
        )
        .bench("EarlyBreak", |b| {
            let mut step = 0;
            while b.keep_running() {
                step += 1;
                if step == 2 {
                    break;
                }
            }
        })
        .run_to(Vec::new())
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        // Calling `reset_timer()` inside `keep_running()` aborts immediately with `InvalidInput`
        // rather than hanging or running `N + 1` iterations.
        let mut in_loop_iters = 0usize;
        let reset_err = Suite::with_config(
            "pkg",
            BenchConfig {
                benchtime: BenchTime::Iterations(10),
                ..BenchConfig::default()
            },
        )
        .bench("ResetInLoop", |b| {
            while b.keep_running() {
                in_loop_iters += 1;
                b.reset_timer();
            }
        })
        .run_to(Vec::new())
        .unwrap_err();
        assert_eq!(reset_err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(in_loop_iters, 1);

        // `--list`, `--help`, `--test`, `--ignored`, `--exact`, `-bench '^$'`, and `with_header`.
        let mut list_out = Vec::new();
        Suite::with_config(
            "pkg",
            BenchConfig {
                filter: Some("Alpha".to_owned()),
                exact: true,
                list_only: true,
                ..BenchConfig::default()
            },
        )
        .bench("Alpha", |b| b.iter(|| 1))
        .bench("AlphaExtra", |b| b.iter(|| 1))
        .run_to(&mut list_out)
        .unwrap();
        assert_eq!(
            String::from_utf8(list_out).unwrap(),
            "BenchmarkAlpha: benchmark\n"
        );

        let mut help_out = Vec::new();
        Suite::with_config(
            "pkg",
            BenchConfig {
                show_help: true,
                ..BenchConfig::default()
            },
        )
        .bench("Alpha", |b| b.iter(|| 1))
        .run_to(&mut help_out)
        .unwrap();
        assert!(String::from_utf8(help_out).unwrap().contains("-benchtime"));

        let mut smoke_runs = 0usize;
        let mut smoke_out = Vec::new();
        Suite::with_config(
            "default_pkg",
            BenchConfig {
                test_mode: true,
                benchtime: BenchTime::Iterations(100),
                count: 5,
                pkg_override: Some("override_pkg".to_owned()),
                ..BenchConfig::default()
            },
        )
        .with_header(BenchHeader {
            goos: "linux".to_owned(),
            goarch: "amd64".to_owned(),
            pkg: "header_pkg".to_owned(),
            cpu: None,
            extra: Vec::new(),
        })
        .bench("Smoke", |b| {
            while b.keep_running() {
                smoke_runs += 1;
            }
        })
        .run_to(&mut smoke_out)
        .unwrap();
        assert_eq!(smoke_runs, 1);
        let smoke_text = String::from_utf8(smoke_out).unwrap();
        assert!(smoke_text.contains("pkg: override_pkg\n"), "{smoke_text}");
        assert!(smoke_text.contains("BenchmarkSmoke"), "{smoke_text}");

        for cfg in [
            BenchConfig {
                run_ignored_only: true,
                ..BenchConfig::default()
            },
            BenchConfig {
                filter: Some("^$".to_owned()),
                ..BenchConfig::default()
            },
        ] {
            let mut empty_out = Vec::new();
            Suite::with_config("pkg", cfg)
                .bench("Alpha", |b| b.iter(|| 1))
                .run_to(&mut empty_out)
                .unwrap();
            let empty_text = String::from_utf8(empty_out).unwrap();
            assert!(!empty_text.contains("BenchmarkAlpha"), "{empty_text}");
            assert!(empty_text.ends_with("PASS\n"), "{empty_text}");
        }
    }

    #[test]
    fn counting_alloc_isolates_threads_and_handles_zeroed_and_shrink() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // NOTE: Synchronize with `AtomicBool` rather than `std::sync::Barrier` between snapshots
        // because `Barrier::wait` lazily heap-allocates `pthread_mutex_t` and `pthread_cond_t` on
        // first use on macOS.
        let start = std::sync::Arc::new(AtomicBool::new(false));
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let start_bg = std::sync::Arc::clone(&start);
        let done_bg = std::sync::Arc::clone(&done);
        let handle = thread::spawn(move || {
            while !start_bg.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            for _ in 0..100 {
                black_box(vec![9u8; 4096]);
            }
            done_bg.store(true, Ordering::Release);
        });
        let before = TEST_ALLOC.snapshot();
        start.store(true, Ordering::Release);
        while !done.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        assert_eq!(
            before.delta_to(TEST_ALLOC.snapshot()),
            AllocSnapshot::default()
        );
        handle.join().unwrap();

        // `vec![0u8; 512]` exercises `alloc_zeroed`, and `reserve_exact` exercises `realloc`
        // growth deterministically even when the system allocator extends in-place.
        let mut out = Vec::new();
        Suite::with_config(
            "pkg",
            BenchConfig {
                benchtime: BenchTime::Iterations(20),
                cpu_procs: 1,
                benchmem: true,
                ..BenchConfig::default()
            },
        )
        .with_alloc(&TEST_ALLOC)
        .bench("ZeroedAndReallocGrow", |b| {
            while b.keep_running() {
                black_box(vec![0u8; 512]);
                let mut v = Vec::<u8>::with_capacity(64);
                v.reserve_exact(192);
                black_box(v);
            }
        })
        .run_to(&mut out)
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("     768 B/op\t       3 allocs/op"), "{text}");

        // Wrapping delta behavior.
        let wrapped = AllocSnapshot {
            allocs: u64::MAX,
            bytes: u64::MAX - 9,
        }
        .delta_to(AllocSnapshot {
            allocs: 2,
            bytes: 10,
        });
        assert_eq!(
            wrapped,
            AllocSnapshot {
                allocs: 3,
                bytes: 20
            }
        );
    }

    #[test]
    fn bencher_drop_stops_active_profiler_on_panic() {
        #[derive(Default)]
        struct PanicProfiler {
            stopped: bool,
        }
        impl Profiler for PanicProfiler {
            fn start_bench(&mut self, _bench_name: &str) -> io::Result<()> {
                Ok(())
            }
            fn stop_bench(&mut self, _bench_name: &str) -> io::Result<()> {
                self.stopped = true;
                Ok(())
            }
        }

        let mut prof = PanicProfiler::default();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Suite::with_config(
                "pkg",
                BenchConfig {
                    benchtime: BenchTime::Iterations(5),
                    ..BenchConfig::default()
                },
            )
            .with_profiler(&mut prof)
            .bench("Boom", |b| {
                let mut step = 0;
                while b.keep_running() {
                    step += 1;
                    if step == 2 {
                        panic!("simulated benchmark panic");
                    }
                }
            })
            .run_to(Vec::new());
        }));
        assert!(res.is_err());
        assert!(
            prof.stopped,
            "expected Bencher::drop to stop active profiler"
        );
    }

    #[test]
    fn custom_profiler_is_scoped_outside_alloc_and_timing_baselines() {
        #[derive(Default)]
        struct RecordingProfiler {
            events: Vec<String>,
            scratch: Vec<Vec<u8>>,
        }

        impl Profiler for RecordingProfiler {
            fn start_bench(&mut self, bench_name: &str) -> io::Result<()> {
                self.scratch.push(vec![0u8; 8192]);
                self.events.push(format!("start:{bench_name}"));
                Ok(())
            }

            fn stop_bench(&mut self, bench_name: &str) -> io::Result<()> {
                self.scratch.push(vec![0u8; 8192]);
                self.events.push(format!("stop:{bench_name}"));
                Ok(())
            }

            fn finish(&mut self) -> io::Result<()> {
                self.events.push("finish".to_owned());
                Ok(())
            }
        }

        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(10),
            count: 2,
            cpu_procs: 1,
            benchmem: true,
            ..BenchConfig::default()
        };
        let mut profiler = RecordingProfiler::default();
        let mut out = Vec::new();
        Suite::with_config("pkg", cfg)
            .with_alloc(&TEST_ALLOC)
            .with_profiler(&mut profiler)
            .bench("ScopedWork", |b| {
                // Allocate 4 KiB outside `keep_running`; neither this nor the profiler's 8 KiB
                // vectors should appear in `B/op`.
                let setup = vec![7u8; 4096];
                while b.keep_running() {
                    let work = vec![setup[0]; 256];
                    black_box(work);
                }
            })
            .run_to(&mut out)
            .unwrap();

        assert_eq!(
            profiler.events,
            vec![
                "start:ScopedWork",
                "stop:ScopedWork",
                "start:ScopedWork",
                "stop:ScopedWork",
                "finish",
            ]
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("     256 B/op\t       1 allocs/op"), "{text}");
    }

    #[test]
    fn report_metric_supports_custom_and_builtin_overrides_without_alloc_noise() {
        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(10),
            count: 1,
            cpu_procs: 2,
            benchmem: true,
            ..BenchConfig::default()
        };
        let mut out = Vec::new();
        Suite::with_config("pkg", cfg)
            .with_alloc(&TEST_ALLOC)
            .bench("Encode/codec=zstd", |b| {
                b.set_bytes(1024);
                let mut total_wire = 0usize;
                while b.keep_running() {
                    let payload = vec![1u8; 64];
                    total_wire += payload.len();
                    // Calling `report_metric` inside `keep_running()` must not add heap allocations
                    // to `B/op` or `allocs/op`, and later calls overwrite earlier ones.
                    b.report_metric(999_999.0, "wire-B/op");
                    b.report_metric(0.12346, "compression-ratio");
                    black_box(payload);
                }
                b.report_metric((total_wire as f64) / (b.iterations() as f64), "wire-B/op");
            })
            .bench("BuiltinOverrides", |b| {
                b.set_bytes(1024);
                b.iter(|| 1);
                // Override built-in `ns/op` (suppressing it with `0.0`), `MB/s` (suppressing with
                // `0.0`), `B/op`, and `allocs/op`, while reporting a custom unit.
                b.report_metric(0.0, "ns/op");
                b.report_metric(0.0, "MB/s");
                b.report_metric(512.0, "B/op");
                b.report_metric(4.0, "allocs/op");
                b.report_metric(-0.0, "z-custom/op");
                b.report_metric(12.346, "a-custom/op");
            })
            .run_to(&mut out)
            .unwrap();

        let text = String::from_utf8(out).unwrap();
        // Custom metrics are sorted lexicographically by unit (`compression-ratio` before
        // `wire-B/op`) and formatted with Go's 4-sig-fig `prettyPrint` widths.
        assert!(
            text.contains(
                "      64 B/op\t       1 allocs/op\t         0.1235 compression-ratio\t        64.00 wire-B/op\n"
            ),
            "{text}"
        );
        // Built-in `ns/op` and `MB/s` were suppressed with `0.0`, `B/op` and `allocs/op` were
        // overridden, and `-0.0` was normalized to `0`.
        assert!(
            text.contains(
                "BenchmarkBuiltinOverrides-2\t      10\t     512 B/op\t       4 allocs/op\t        12.35 a-custom/op\t         0 z-custom/op\n"
            ),
            "{text}"
        );

        // Invalid units (empty, whitespace, control characters) and invalid values (`NaN`, `inf`,
        // negative built-in overrides) fail with `InvalidInput`.
        for (unit, val) in [
            ("", 1.0),
            ("wire B/op", 1.0),
            ("wire\tB/op", 1.0),
            ("wire\nB/op", 1.0),
            ("wire-B/op", f64::NAN),
            ("wire-B/op", f64::INFINITY),
            ("ns/op", -1.0),
            ("MB/s", -1.0),
            ("B/op", -1.0),
            ("allocs/op", -1.0),
        ] {
            let err = Suite::with_config(
                "pkg",
                BenchConfig {
                    benchtime: BenchTime::Iterations(5),
                    ..BenchConfig::default()
                },
            )
            .bench("InvalidMetric", |b| {
                while b.keep_running() {
                    b.report_metric(val, unit);
                }
            })
            .run_to(Vec::new())
            .unwrap_err();
            assert_eq!(
                err.kind(),
                io::ErrorKind::InvalidInput,
                "expected InvalidInput for ({unit:?}, {val})"
            );
        }

        // Suppressing `ns/op` with `0.0` when no other metrics are emitted is rejected so an empty
        // measurement line cannot be produced.
        let empty_line_err = Suite::with_config(
            "pkg",
            BenchConfig {
                benchtime: BenchTime::Iterations(1),
                benchmem: false,
                ..BenchConfig::default()
            },
        )
        .bench("NoMetricsLeft", |b| {
            b.iter(|| 1);
            b.report_metric(0.0, "ns/op");
        })
        .run_to(Vec::new())
        .unwrap_err();
        assert_eq!(empty_line_err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(not(feature = "pprof"))]
    #[test]
    fn rejects_profile_flags_when_pprof_feature_is_disabled() {
        let cfg = BenchConfig {
            benchtime: BenchTime::Iterations(1),
            mem_profile: Some(PathBuf::from("mem.pb.gz")),
            ..BenchConfig::default()
        };
        let mut out = Vec::new();
        let err = Suite::with_config("pkg", cfg)
            .bench("Noop", |b| b.iter(|| 1))
            .run_to(&mut out)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }
}
