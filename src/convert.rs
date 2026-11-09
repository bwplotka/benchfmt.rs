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

//! Converters from Criterion directory trees and Rust `libtest` output into Go `benchfmt`.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::json::parse_json;
use crate::{BenchHeader, BenchRecord, sanitize_bench_name};

/// Options controlling conversion from Criterion or `libtest` into `benchfmt`.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// Metadata header emitted at the top of the `benchfmt` stream.
    pub header: BenchHeader,
    /// Optional `-N` GOMAXPROCS/CPU suffix appended to benchmark names (for example `Some(2)`).
    pub cpu_procs: Option<usize>,
    /// Number of contiguous sample buckets to aggregate Criterion samples into (`0` emits every
    /// raw sample; `6` aggregates Criterion's 100 samples into 6 equal epochs matching `-count 6`).
    pub criterion_epochs: usize,
    /// Which Criterion baseline directory name to read (defaults to `"new"`).
    pub criterion_baseline: String,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            header: BenchHeader::detect("rust"),
            cpu_procs: None,
            criterion_epochs: 6,
            criterion_baseline: "new".to_owned(),
        }
    }
}

/// Walks a Criterion output directory (typically `target/criterion`) and writes Go `benchfmt`
/// lines to `out`.
pub fn convert_criterion_dir(
    dir: &Path,
    opts: &ConvertOptions,
    mut out: impl Write,
) -> io::Result<usize> {
    let mut bench_files = Vec::new();
    collect_criterion_benchmarks(dir, &opts.criterion_baseline, &mut bench_files)?;
    bench_files.sort();

    opts.header.write(&mut out)?;
    let mut emitted = 0usize;

    for bench_json_path in bench_files {
        let Some(parent) = bench_json_path.parent() else {
            continue;
        };
        let sample_json_path = parent.join("sample.json");
        if !sample_json_path.exists() {
            continue;
        }
        let bench_raw = fs::read_to_string(&bench_json_path)?;
        let sample_raw = fs::read_to_string(&sample_json_path)?;

        let bench_val =
            parse_json(&bench_raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let sample_val =
            parse_json(&sample_raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let Some(bench_obj) = bench_val.as_object() else {
            continue;
        };
        let Some(sample_obj) = sample_val.as_object() else {
            continue;
        };

        let full_id = bench_obj
            .get("full_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let bytes_per_op = bench_obj
            .get("throughput")
            .and_then(|v| v.as_object())
            .and_then(|tp| {
                tp.get("Bytes")
                    .or_else(|| tp.get("BytesDecimal"))
                    .and_then(|n| n.as_f64())
                    .filter(|&n| n.is_finite() && n > 0.0)
                    .map(|n| n as u64)
            });

        let Some(iters) = sample_obj.get("iters").and_then(|v| v.as_array()) else {
            continue;
        };
        let Some(times) = sample_obj.get("times").and_then(|v| v.as_array()) else {
            continue;
        };
        if iters.len() != times.len() || iters.is_empty() {
            continue;
        }

        let pairs: Vec<(f64, f64)> = iters
            .iter()
            .zip(times.iter())
            .filter_map(|(i, t)| Some((i.as_f64()?, t.as_f64()?)))
            .filter(|&(i, t)| i.is_finite() && t.is_finite() && i > 0.0 && t >= 0.0)
            .collect();
        if pairs.is_empty() {
            continue;
        }

        let name = sanitize_bench_name(full_id);
        let bucket_count = if opts.criterion_epochs == 0 {
            pairs.len()
        } else {
            opts.criterion_epochs.min(pairs.len())
        };

        for bucket_idx in 0..bucket_count {
            let start = bucket_idx * pairs.len() / bucket_count;
            let end = (bucket_idx + 1) * pairs.len() / bucket_count;
            let slice = &pairs[start..end];
            let sum_iters: f64 = slice.iter().map(|(i, _)| *i).sum();
            let sum_ns: f64 = slice.iter().map(|(_, t)| *t).sum();
            if !sum_iters.is_finite() || !sum_ns.is_finite() || sum_iters <= 0.0 {
                continue;
            }
            let ns_per_op = sum_ns / sum_iters;
            if !ns_per_op.is_finite() || ns_per_op < 0.0 {
                continue;
            }
            let mb_per_sec = bytes_per_op.and_then(|b| {
                if sum_ns > 0.0 {
                    // Match Go testing.B: decimal MB/s = (bytes * iters / 1e6) / (ns / 1e9).
                    Some((b as f64 * sum_iters * 1_000.0) / sum_ns)
                } else {
                    None
                }
            });
            let rec = BenchRecord {
                name: &name,
                cpu_procs: opts.cpu_procs,
                iterations: (sum_iters.round() as u64).clamp(1, i64::MAX as u64),
                ns_per_op,
                mb_per_sec,
                alloc_bytes_per_op: None,
                allocs_per_op: None,
                extra_metrics: Vec::new(),
            };
            rec.write(&mut out)?;
            emitted += 1;
        }
    }

    writeln!(out, "PASS")?;
    Ok(emitted)
}

fn collect_criterion_benchmarks(
    dir: &Path,
    baseline: &str,
    out: &mut Vec<PathBuf>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        // Do not follow directory symlinks so symlink loops cannot cause unbounded recursion.
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        if path.file_name().and_then(|s| s.to_str()) == Some(baseline) {
            let candidate = path.join("benchmark.json");
            if candidate.is_file() {
                out.push(candidate);
                continue;
            }
        }
        collect_criterion_benchmarks(&path, baseline, out)?;
    }
    Ok(())
}

/// Converts Rust nightly `libtest` benchmark lines
/// (`test foo::bar ... bench:   1,234.56 ns/iter (+/- 12.34) = 456 MB/s`)
/// into Go `benchfmt` records.
pub fn convert_libtest_output(
    input: &str,
    opts: &ConvertOptions,
    mut out: impl Write,
) -> io::Result<usize> {
    opts.header.write(&mut out)?;
    let mut emitted = 0usize;

    for line in input.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("test ") else {
            continue;
        };
        let Some((raw_name, bench_part)) = rest.split_once("... bench:") else {
            continue;
        };
        let name = sanitize_bench_name(raw_name.trim());
        let bench_part = bench_part.trim();
        let Some((ns_str, after_ns)) = bench_part.split_once("ns/iter") else {
            continue;
        };
        let cleaned_ns: String = ns_str.trim().chars().filter(|&c| c != ',').collect();
        let Ok(ns_per_op) = cleaned_ns.parse::<f64>() else {
            continue;
        };
        if !ns_per_op.is_finite() || ns_per_op < 0.0 {
            continue;
        }
        let mb_per_sec = after_ns
            .split_once('=')
            .and_then(|(_, rhs)| rhs.trim().strip_suffix("MB/s"))
            .and_then(|s| {
                let cleaned: String = s.trim().chars().filter(|&c| c != ',').collect();
                cleaned.parse::<f64>().ok()
            })
            .filter(|&v| v.is_finite() && v > 0.0);

        let rec = BenchRecord {
            name: &name,
            cpu_procs: opts.cpu_procs,
            iterations: 1,
            ns_per_op,
            mb_per_sec,
            alloc_bytes_per_op: None,
            allocs_per_op: None,
            extra_metrics: Vec::new(),
        };
        rec.write(&mut out)?;
        emitted += 1;
    }

    writeln!(out, "PASS")?;
    Ok(emitted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn criterion_supports_raw_epochs_bytes_decimal_and_report_named_groups() {
        let tmp =
            std::env::temp_dir().join(format!("benchfmt_convert_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // Benchmark group literally named `report` containing both Criterion's HTML
        // `report/index.html` and a nested `report/case_1/base` sub-benchmark.
        let group_dir = tmp.join("report/case_1/base");
        let html_dir = tmp.join("report");
        fs::create_dir_all(&group_dir).unwrap();
        fs::write(html_dir.join("index.html"), "<html></html>").unwrap();

        fs::write(
            group_dir.join("benchmark.json"),
            r#"{"full_id":"report/case=1","throughput":{"BytesDecimal":500}}"#,
        )
        .unwrap();
        fs::write(
            group_dir.join("sample.json"),
            r#"{"iters":[1.0,2.0,3.0],"times":[10.0,20.0,30.0]}"#,
        )
        .unwrap();

        #[cfg(unix)]
        std::os::unix::fs::symlink(&tmp, tmp.join("loop_link")).unwrap();

        let opts = ConvertOptions {
            header: BenchHeader {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                pkg: "pkg".to_owned(),
                cpu: None,
                extra: Vec::new(),
            },
            cpu_procs: None,
            criterion_epochs: 0,
            criterion_baseline: "base".to_owned(),
        };
        let mut out = Vec::new();
        let emitted = convert_criterion_dir(&tmp, &opts, &mut out).unwrap();
        let _ = fs::remove_dir_all(&tmp);

        assert_eq!(emitted, 3);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("BenchmarkReport/case=1\t"), "{text}");
        assert!(text.contains("50000.00 MB/s"), "{text}");

        let missing_err = convert_criterion_dir(&tmp, &opts, Vec::new()).unwrap_err();
        assert_eq!(missing_err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn libtest_handles_missing_throughput_and_ignores_non_finite_lines() {
        let sample = "\
test ok::plain ... bench:       9.50 ns/iter (+/- 0.20)
test bad::nan ... bench:       NaN ns/iter (+/- 0.00)
test ok::comma_tp ... bench:   2,000 ns/iter (+/- 10) = 1,234.50 MB/s
";
        let mut out = Vec::new();
        let count = convert_libtest_output(sample, &ConvertOptions::default(), &mut out).unwrap();
        assert_eq!(count, 2);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("BenchmarkOk::plain"), "{text}");
        assert!(!text.contains("Bad::nan"), "{text}");
        assert!(text.contains("1234.50 MB/s"), "{text}");
    }
}
