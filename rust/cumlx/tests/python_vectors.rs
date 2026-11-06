//! The Rust metrics against what the Python metrics returned for the same inputs.
//!
//! `scripts/rust_vectors.py` writes `tests/data/python-vectors.json` from `src/cumlx/verify/metrics.py` and
//! `weights.py`, and `tests/test_rust_parity.py` fails when that file is stale. Here every Rust result must
//! have the same keys in the same order, equal strings, integers and nulls, and floats within 1e-12 relative.

use cumlx::errors::{ExitCode, ERROR_CODES};
use cumlx::metrics::{
    compare, compare_kernel_output, noise_thresholds, resolve_threshold, Array, KernelLimits,
    Limits, COMPOSITE_NOISE_FACTOR, FLOAT32_TINY, KERNEL_THRESHOLDS, LEAF_NOISE_FACTOR,
    PORT_WEIGHTS_RELL2, THRESHOLD_PROFILES,
};
use cumlx::safetensors::{Dtype, Tensor};
use cumlx::weights::{
    compare_parameter, Nd, QUANTIZED_MAX_SCALE_ERROR, QUANTIZED_MIN_COSINE, REL_L2_LIMITS,
    TIED_EMBEDDING,
};
use serde_json::{Map, Value};

const REL_TOL: f64 = 1e-12;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/python-vectors.json"
    );
    let text = std::fs::read_to_string(path).expect("run python scripts/rust_vectors.py");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["schema"], "cumlx.rust-vectors.v1");
    v
}

fn differences(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if x.is_f64() || y.is_f64() {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                if !(x == y || (x - y).abs() <= REL_TOL * x.abs().max(y.abs())) {
                    out.push(format!("{path}: {x:e} != {y:e}"));
                }
            } else if x != y {
                out.push(format!("{path}: {x} != {y}"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (Vec<&String>, Vec<&String>) = (x.keys().collect(), y.keys().collect());
            if kx != ky {
                out.push(format!("{path}: keys {kx:?} != {ky:?}"));
                return;
            }
            for k in x.keys() {
                differences(&x[k], &y[k], &format!("{path}.{k}"), out);
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                out.push(format!("{path}: length {} != {}", x.len(), y.len()));
                return;
            }
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                differences(p, q, &format!("{path}[{i}]"), out);
            }
        }
        _ => {
            if a != b {
                out.push(format!("{path}: {a} != {b}"));
            }
        }
    }
}

fn assert_same(rust: &Value, python: &Value, what: &str) {
    let mut out = Vec::new();
    differences(rust, python, "$", &mut out);
    assert!(
        out.is_empty(),
        "{what}: Rust differs from Python:\n{}",
        out.join("\n")
    );
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn tensor(v: &Value) -> Tensor {
    Tensor {
        dtype: Dtype::from_tag(v["dtype"].as_str().unwrap()).unwrap(),
        shape: v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_u64().unwrap() as usize)
            .collect(),
        raw: hex(v["hex"].as_str().unwrap()),
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn constants_and_code_tables_equal_python() {
    let v = vectors();
    let c = &v["constants"];
    let profiles = c["threshold_profiles"].as_object().unwrap();
    assert_eq!(profiles.len(), THRESHOLD_PROFILES.len());
    for (p, (name, py)) in THRESHOLD_PROFILES.iter().zip(profiles) {
        assert_eq!(p.name, name);
        assert_eq!(
            (
                p.rel_l2,
                p.isolated_rel_l2,
                p.composite_rel_l2_max,
                p.zero_atol
            ),
            (
                f(&py["rel_l2"]),
                f(&py["isolated_rel_l2"]),
                f(&py["composite_rel_l2_max"]),
                f(&py["zero_atol"])
            )
        );
    }
    assert_eq!(PORT_WEIGHTS_RELL2, f(&c["port_weights_rel_l2"]));
    assert_eq!(LEAF_NOISE_FACTOR, f(&c["leaf_noise_factor"]));
    assert_eq!(COMPOSITE_NOISE_FACTOR, f(&c["composite_noise_factor"]));
    let k = &c["kernel_thresholds"];
    assert_eq!(
        KERNEL_THRESHOLDS,
        KernelLimits {
            rel_l2: f(&k["rel_l2"]),
            max_rel: f(&k["max_rel"]),
            zero_atol: f(&k["zero_atol"])
        }
    );
    assert_eq!(FLOAT32_TINY, f(&c["float32_tiny"]));
    let limits = c["weights_rel_l2_limits"].as_object().unwrap();
    assert_eq!(REL_L2_LIMITS.len(), limits.len());
    for ((name, value), (py_name, py_value)) in REL_L2_LIMITS.iter().zip(limits) {
        assert_eq!((*name, *value), (py_name.as_str(), f(py_value)));
    }
    assert_eq!(QUANTIZED_MIN_COSINE, f(&c["quantized_min_cosine"]));
    assert_eq!(
        QUANTIZED_MAX_SCALE_ERROR,
        f(&c["quantized_max_scale_error"])
    );
    assert_eq!(TIED_EMBEDDING, c["tied_embedding"]);
    let codes: Vec<(String, i64, String)> = c["error_codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_str().unwrap().into(),
                r[1].as_i64().unwrap(),
                r[2].as_str().unwrap().into(),
            )
        })
        .collect();
    let ours: Vec<(String, i64, String)> = ERROR_CODES
        .iter()
        .map(|(code, exit, text)| (code.to_string(), exit.code() as i64, text.to_string()))
        .collect();
    assert_eq!(ours, codes, "ERROR_CODES differ between Rust and Python");
    let exits: Vec<(i64, String, String)> = c["exit_codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_i64().unwrap(),
                r[1].as_str().unwrap().into(),
                r[2].as_str().unwrap().into(),
            )
        })
        .collect();
    let ours: Vec<(i64, String, String)> = ExitCode::ALL
        .iter()
        .map(|e| (e.code() as i64, e.name().into(), e.meaning().into()))
        .collect();
    assert_eq!(ours, exits, "exit codes differ between Rust and Python");
}

#[test]
fn thresholds_equal_python() {
    let v = vectors();
    let resolved = v["thresholds"]["resolved"].as_array().unwrap();
    assert!(resolved.len() >= 16);
    for case in resolved {
        let profile = case["profile"].as_str().unwrap();
        let rel = case["rel_l2"].as_f64();
        let pwg = case["port_weights_golden"].as_bool().unwrap();
        let t = resolve_threshold(profile, rel, pwg).unwrap();
        assert_same(
            &t.to_value(),
            &case["result"],
            &format!("resolve_threshold({profile}, {rel:?}, {pwg})"),
        );
        for n in case["noise"].as_array().unwrap() {
            let (leaf, composite) = noise_thresholds(&t, f(&n["median"]));
            assert_same(
                &serde_json::json!([leaf, composite]),
                &n["result"],
                &format!("noise_thresholds {profile} {n}"),
            );
        }
    }
    for case in v["thresholds"]["errors"].as_array().unwrap() {
        let rel = if case["infinite"].as_bool().unwrap() {
            Some(f64::INFINITY)
        } else {
            case["rel_l2"].as_f64()
        };
        let err = resolve_threshold(case["profile"].as_str().unwrap(), rel, false).unwrap_err();
        assert_eq!(err, case["error"].as_str().unwrap());
    }
}

#[test]
fn compare_equals_python() {
    let v = vectors();
    let cases = v["compare"].as_array().unwrap();
    assert!(cases.len() >= 20);
    for case in cases {
        let a = Array::from_tensor(&tensor(&case["actual"]));
        let e = Array::from_tensor(&tensor(&case["expected"]));
        let lim = Limits {
            rel_l2: f(&case["limits"]["rel_l2"]),
            zero_atol: f(&case["limits"]["zero_atol"]),
        };
        assert_same(
            &Value::Object(compare(&a, &e, lim)),
            &case["result"],
            case["name"].as_str().unwrap(),
        );
    }
}

#[test]
fn compare_kernel_output_equals_python() {
    let v = vectors();
    let cases = v["compare_kernel_output"].as_array().unwrap();
    assert!(cases.len() >= 15);
    for case in cases {
        let limits = case["limits"]
            .as_object()
            .map(|l: &Map<String, Value>| KernelLimits {
                rel_l2: f(&l["rel_l2"]),
                max_rel: f(&l["max_rel"]),
                zero_atol: f(&l["zero_atol"]),
            });
        let got =
            compare_kernel_output(&tensor(&case["actual"]), &tensor(&case["expected"]), limits);
        assert_same(
            &Value::Object(got),
            &case["result"],
            case["name"].as_str().unwrap(),
        );
    }
}

#[test]
fn weight_comparison_equals_python() {
    let v = vectors();
    let cases = v["weights_compare"].as_array().unwrap();
    assert!(cases.len() >= 20);
    for case in cases {
        let port = tensor(&case["port"]);
        let orig = tensor(&case["original"]);
        // Python reads every port parameter as float32.
        let port = Nd {
            shape: port.shape.clone(),
            values: port.to_f64().into_iter().map(|x| x as f32 as f64).collect(),
        };
        let orig = Nd {
            shape: orig.shape.clone(),
            values: orig.to_f64(),
        };
        let got = compare_parameter(&port, &orig, case["kind"].as_str().unwrap());
        assert_same(
            &Value::Object(got),
            &case["result"],
            case["name"].as_str().unwrap(),
        );
    }
}
