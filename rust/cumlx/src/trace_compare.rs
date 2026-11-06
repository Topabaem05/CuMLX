//! Compare two golden traces of one model, such as a CUDA capture and a CPU capture
//! (`src/cumlx/verify/trace_compare.py`).
//!
//! This measures how much the reference itself moves between devices, or between two runs on one device. If
//! that number is above a threshold, a port can fail for a reason that has nothing to do with the port.

use crate::envelope::num;
use crate::errors::{CumlxError, CumlxResult};
use crate::metrics::{compare, Array, Limits};
use crate::pyfmt::py_str;
use crate::safetensors::Tensor;
use crate::trace::Trace;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::Path;

/// Schema of the comparison report.
pub const SCHEMA: &str = "cumlx.trace-compare.v1";
/// The float32 threshold of `cumlx diff`.
pub const DEFAULT_REL_L2: f64 = 1e-4;
/// Length of the `worst` list.
pub const WORST_LIMIT: usize = 20;

/// The report and every row.
#[derive(Clone, Debug)]
pub struct TraceComparison {
    /// The dict that Python's `compare_traces` returns.
    pub report: Map<String, Value>,
    /// One row per record of the first trace: the metrics, or `verdict: missing`.
    pub rows: Vec<Map<String, Value>>,
}

/// `np.array_equal(a, b)`: the same shape and equal values (NaN is never equal).
pub fn array_equal(a: &Tensor, b: &Tensor) -> bool {
    if a.shape != b.shape {
        return false;
    }
    match (a.to_i128(), b.to_i128()) {
        (Some(x), Some(y)) => x == y,
        _ => a.to_f64() == b.to_f64(),
    }
}

fn input_invalid(message: String) -> CumlxError {
    CumlxError::new("E_INPUT_INVALID", message).hint("capture both with the same inputs")
}

/// `compare_traces(a, b, rel_l2=...)`.
pub fn compare_traces(a: &str, b: &str, rel_l2: f64) -> CumlxResult<TraceComparison> {
    let first = Trace::open(Path::new(a), true)?;
    let second = Trace::open(Path::new(b), true)?;
    let limits = Limits {
        rel_l2,
        zero_atol: rel_l2,
    };
    let (sa_all, sb_all) = (&first.manifest.samples, &second.manifest.samples);
    if sa_all.len() != sb_all.len() {
        return Err(input_invalid(
            "the traces have a different number of samples".into(),
        ));
    }
    let mut rows: Vec<Map<String, Value>> = Vec::new();
    for (i, (sa, sb)) in sa_all.iter().zip(sb_all).enumerate() {
        if !array_equal(&first.token_ids(i)?, &second.token_ids(i)?) {
            return Err(input_invalid(format!(
                "sample {} has different token ids in the two traces",
                sa.id_text()
            )));
        }
        let other: HashMap<String, &crate::trace::ModuleRecord> =
            sb.records.iter().map(|r| (r.key_text(), r)).collect();
        for r in &sa.records {
            let mut row = Map::new();
            row.insert("sample".into(), sa.id.clone());
            row.insert("key".into(), r.key.clone());
            let twin = other.get(&r.key_text());
            let (Some(twin), Some(out)) = (twin, r.output.as_ref()) else {
                row.insert("verdict".into(), "missing".into());
                rows.push(row);
                continue;
            };
            let Some(twin_out) = twin.output.as_ref() else {
                row.insert("verdict".into(), "missing".into());
                rows.push(row);
                continue;
            };
            row.insert("type".into(), r.r#type.clone());
            row.insert("phase".into(), r.phase.clone());
            let actual = Array::from_tensor(&first.tensor(i, &out.tensor)?);
            let expected = Array::from_tensor(&second.tensor(i, &twin_out.tensor)?);
            row.extend(compare(&actual, &expected, limits));
            rows.push(row);
        }
    }
    let rel = |x: &Map<String, Value>| x.get("rel_l2").and_then(Value::as_f64);
    let mut values: Vec<f64> = rows.iter().filter_map(rel).collect();
    values.sort_by(|x, y| y.total_cmp(x));
    let failed = rows
        .iter()
        .filter(|x| x.get("verdict").and_then(Value::as_str) == Some("fail"))
        .count();
    let missing = rows
        .iter()
        .filter(|x| x.get("verdict").and_then(Value::as_str) == Some("missing"))
        .count();
    let verdict = if failed > 0 {
        "diverged"
    } else if missing > 0 || values.is_empty() {
        "incomplete"
    } else {
        "ok"
    };
    let mut worst: Vec<&Map<String, Value>> = rows.iter().filter(|x| rel(x).is_some()).collect();
    worst.sort_by(|x, y| rel(y).unwrap().total_cmp(&rel(x).unwrap()));
    let side = |path: &str, t: &Trace| {
        let mut m = Map::new();
        m.insert("path".into(), path.into());
        m.insert("source".into(), t.manifest.source.clone());
        Value::Object(m)
    };
    let mut summary = Map::new();
    summary.insert("records".into(), rows.len().into());
    summary.insert("compared".into(), values.len().into());
    summary.insert("above_threshold".into(), failed.into());
    summary.insert("missing".into(), missing.into());
    summary.insert(
        "worst_rel_l2".into(),
        values.first().map(|v| num(*v)).unwrap_or(Value::Null),
    );
    summary.insert(
        "median_rel_l2".into(),
        values
            .get(values.len() / 2)
            .map(|v| num(*v))
            .unwrap_or(Value::Null),
    );
    let mut report = Map::new();
    report.insert("schema".into(), SCHEMA.into());
    report.insert("verdict".into(), verdict.into());
    report.insert("a".into(), side(a, &first));
    report.insert("b".into(), side(b, &second));
    report.insert("threshold_rel_l2".into(), num(rel_l2));
    report.insert("summary".into(), Value::Object(summary));
    report.insert(
        "worst".into(),
        Value::Array(
            worst
                .into_iter()
                .take(WORST_LIMIT)
                .map(|x| Value::Object(x.clone()))
                .collect(),
        ),
    );
    Ok(TraceComparison { report, rows })
}

/// The one-line summary of `cumlx tracediff`. `rel_l2` is the threshold as given, which may be infinite.
pub fn summary_line(report: &Map<String, Value>, rel_l2: f64) -> String {
    let s = report["summary"].as_object().expect("summary");
    let worst = &s["worst_rel_l2"];
    let median = &s["median_rel_l2"];
    format!(
        "{}: worst rel_l2 {}, median {}, {} of {} outputs above {}",
        py_str(&report["verdict"]),
        py_str(worst),
        py_str(median),
        py_str(&s["above_threshold"]),
        py_str(&s["compared"]),
        crate::pyfmt::format_g(rel_l2, 6)
    )
}
