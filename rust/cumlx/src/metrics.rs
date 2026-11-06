//! Numerical comparison of a ported output against a golden output, as `src/cumlx/verify/metrics.py`.
//!
//! Every constant here is a copy of the Python one, with the measurement that set it. Do not change one
//! without changing the Python constant and its record; `tests/test_rust_parity.py` compares the two tables.
//!
//! All metrics are computed in float64. `rel_l2` is the primary verdict metric; `max_abs` and `cosine` are
//! reported for diagnosis. Sums use eight partial sums, so the results can differ from NumPy's BLAS sums in
//! the last bits; the parity tests allow a relative difference of 1e-12.

use crate::envelope::num;
use crate::safetensors::Tensor;
use serde_json::{Map, Value};

/// One threshold profile of `THRESHOLD_PROFILES`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    /// Name of the profile.
    pub name: &'static str,
    /// Chain outputs in the native dtype.
    pub rel_l2: f64,
    /// Upper limit for leaf modules in isolated replay.
    pub isolated_rel_l2: f64,
    /// Upper limit for modules that contain other parametrized modules.
    pub composite_rel_l2_max: f64,
    /// Limit of `max_abs` when the golden is all zero.
    pub zero_atol: f64,
}

/// Chosen by the dtype of the PORT's parameters. Measured worst clean values (float32-origin weights):
/// float32 2.8e-5; float16 leaf 5.9e-4, composite 3.5e-3; bfloat16 leaf 4.7e-3, composite 7.5e-2.
/// Measured on Qwen2.5-0.5B and SmolLM2-135M, Apple M5 Pro GPU and CPU, 2026-11-02 (docs/CLI.md).
pub const THRESHOLD_PROFILES: [Profile; 4] = [
    Profile {
        name: "float32",
        rel_l2: 1e-4,
        isolated_rel_l2: 1e-4,
        composite_rel_l2_max: 1e-4,
        zero_atol: 1e-6,
    },
    Profile {
        name: "float16",
        rel_l2: 5e-3,
        isolated_rel_l2: 2e-3,
        composite_rel_l2_max: 1e-2,
        zero_atol: 1e-3,
    },
    Profile {
        name: "bfloat16",
        rel_l2: 3e-2,
        isolated_rel_l2: 1.5e-2,
        composite_rel_l2_max: 2e-1,
        zero_atol: 1e-2,
    },
    Profile {
        name: "quantized",
        rel_l2: 3e-2,
        isolated_rel_l2: 3e-2,
        composite_rel_l2_max: 3e-2,
        zero_atol: 1e-2,
    },
];

/// A golden that holds the port's own weights leaves only float32 arithmetic noise.
pub const PORT_WEIGHTS_RELL2: f64 = 1e-4;
/// Twice the worst measured ratio to the median leaf error, on 10 clean float32-origin ports: leaf max/median 3.0.
pub const LEAF_NOISE_FACTOR: f64 = 6.0;
/// Composite max/median 48 (one Qwen2.5 decoder layer near cancellation), twice that.
pub const COMPOSITE_NOISE_FACTOR: f64 = 100.0;

/// A resolved threshold, as `resolve_threshold` returns it.
#[derive(Clone, Debug, PartialEq)]
pub struct Threshold {
    /// Chain outputs in the native dtype.
    pub rel_l2: f64,
    /// Leaf modules in isolated replay.
    pub isolated_rel_l2: f64,
    /// Composite modules in isolated replay.
    pub composite_rel_l2_max: f64,
    /// Limit of `max_abs` when the golden is all zero.
    pub zero_atol: f64,
    /// Lower bound of the run-dependent thresholds.
    pub noise_floor: f64,
    /// True when `--rel-l2` set every limit.
    pub fixed_by_user: bool,
    /// Name of the profile.
    pub profile: String,
}

impl Threshold {
    /// The dict that Python returns, in its key order.
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("rel_l2".into(), num(self.rel_l2));
        m.insert("isolated_rel_l2".into(), num(self.isolated_rel_l2));
        m.insert(
            "composite_rel_l2_max".into(),
            num(self.composite_rel_l2_max),
        );
        m.insert("zero_atol".into(), num(self.zero_atol));
        m.insert("noise_floor".into(), num(self.noise_floor));
        if self.fixed_by_user {
            m.insert("fixed_by_user".into(), true.into());
        }
        m.insert("profile".into(), self.profile.clone().into());
        Value::Object(m)
    }
}

/// The limits that [`compare`] reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Pass when `rel_l2 <= rel_l2`.
    pub rel_l2: f64,
    /// Pass a zero golden when `max_abs <= zero_atol`.
    pub zero_atol: f64,
}

impl From<&Threshold> for Limits {
    fn from(t: &Threshold) -> Limits {
        Limits {
            rel_l2: t.rel_l2,
            zero_atol: t.zero_atol,
        }
    }
}

/// `resolve_threshold(profile, rel_l2, port_weights_golden=...)`.
pub fn resolve_threshold(
    profile: &str,
    rel_l2: Option<f64>,
    port_weights_golden: bool,
) -> Result<Threshold, String> {
    let Some(p) = THRESHOLD_PROFILES.iter().find(|p| p.name == profile) else {
        return Err(format!(
            "unknown threshold profile {}",
            crate::pyfmt::repr_str(profile)
        ));
    };
    let mut t = Threshold {
        rel_l2: p.rel_l2,
        isolated_rel_l2: p.isolated_rel_l2,
        composite_rel_l2_max: p.composite_rel_l2_max,
        zero_atol: p.zero_atol,
        noise_floor: PORT_WEIGHTS_RELL2,
        fixed_by_user: false,
        profile: profile.into(),
    };
    if port_weights_golden {
        t.isolated_rel_l2 = PORT_WEIGHTS_RELL2;
        t.composite_rel_l2_max = PORT_WEIGHTS_RELL2;
    }
    if let Some(r) = rel_l2 {
        if !(r.is_finite() && r > 0.0) {
            return Err("rel_l2 threshold must be positive and finite".into());
        }
        t.rel_l2 = r;
        t.isolated_rel_l2 = r;
        t.composite_rel_l2_max = r;
        t.fixed_by_user = true;
    }
    Ok(t)
}

/// `(leaf, composite)` isolated thresholds for one run, from its median leaf error.
pub fn noise_thresholds(t: &Threshold, median_leaf_error: f64) -> (f64, f64) {
    let (leaf, composite) = (t.isolated_rel_l2, t.composite_rel_l2_max);
    if t.fixed_by_user {
        return (leaf, composite);
    }
    let floor = t.noise_floor;
    (
        py_min(
            py_max(LEAF_NOISE_FACTOR * median_leaf_error, floor),
            py_max(leaf, floor),
        ),
        py_min(
            py_max(COMPOSITE_NOISE_FACTOR * median_leaf_error, floor),
            py_max(composite, floor),
        ),
    )
}

/// Python's `max(a, b)`: the first argument unless the second is greater (NaN never is).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: the first argument unless the second is smaller.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Sum with eight partial sums: close to a BLAS dot product, and independent of the input length.
pub fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut acc = [0.0f64; 8];
    for (i, v) in values.enumerate() {
        acc[i & 7] += v;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// `float(np.dot(a, b))`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    sum(a.iter().zip(b).map(|(x, y)| x * y))
}

/// `float(np.linalg.norm(a))`.
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn shape_value(shape: &[usize]) -> Value {
    shape.iter().map(|s| Value::from(*s)).collect()
}

/// An array of float64 values with a shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    /// Shape; empty for a scalar.
    pub shape: Vec<usize>,
    /// Values in C order.
    pub values: Vec<f64>,
}

impl Array {
    /// A tensor as float64.
    pub fn from_tensor(t: &Tensor) -> Array {
        Array {
            shape: t.shape.clone(),
            values: t.to_f64(),
        }
    }
}

/// `compare(actual, expected, threshold)`: metrics plus `verdict` in {`pass`, `fail`} and a `reason` for failures.
pub fn compare(actual: &Array, expected: &Array, limits: Limits) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("shape".into(), shape_value(&actual.shape));
    out.insert("expected_shape".into(), shape_value(&expected.shape));
    let none = |out: &mut Map<String, Value>| {
        out.insert("rel_l2".into(), Value::Null);
        out.insert("max_abs".into(), Value::Null);
        out.insert("cosine".into(), Value::Null);
    };
    if actual.shape != expected.shape {
        if actual.values.len() == expected.values.len() && !actual.values.is_empty() {
            out.insert("reshaped".into(), true.into());
        } else {
            out.insert("verdict".into(), "fail".into());
            out.insert("reason".into(), "shape_mismatch".into());
            none(&mut out);
            return out;
        }
    }
    let (a, e) = (&actual.values, &expected.values);
    let bad = a.iter().filter(|x| !x.is_finite()).count();
    if bad > 0 {
        out.insert("verdict".into(), "fail".into());
        out.insert("reason".into(), "nonfinite_output".into());
        out.insert("nonfinite_count".into(), bad.into());
        none(&mut out);
        return out;
    }
    if e.iter().any(|x| !x.is_finite()) {
        out.insert("verdict".into(), "fail".into());
        out.insert("reason".into(), "nonfinite_golden".into());
        none(&mut out);
        return out;
    }
    let diff: Vec<f64> = a.iter().zip(e).map(|(x, y)| x - y).collect();
    let max_abs = diff.iter().fold(0.0f64, |m, d| m.max(d.abs()));
    let (en, an, dn) = (norm(e), norm(a), norm(&diff));
    let rel = if en > 0.0 {
        dn / en
    } else if dn == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let ae = dot(a, e);
    let cos = if an > 0.0 && en > 0.0 {
        ae / (an * en)
    } else if an == en {
        1.0
    } else {
        0.0
    };
    out.insert("rel_l2".into(), num(rel));
    out.insert("max_abs".into(), num(max_abs));
    out.insert("cosine".into(), num(cos));
    out.insert(
        "scale".into(),
        if en > 0.0 {
            num(ae / (en * en))
        } else {
            Value::Null
        },
    );
    if en == 0.0 {
        let ok = max_abs <= limits.zero_atol;
        out.insert("verdict".into(), if ok { "pass" } else { "fail" }.into());
        out.insert("basis".into(), "zero_atol".into());
        if !ok {
            out.insert("reason".into(), "golden_is_zero_but_output_is_not".into());
        }
        return out;
    }
    let ok = rel <= limits.rel_l2;
    out.insert("verdict".into(), if ok { "pass" } else { "fail" }.into());
    out.insert("basis".into(), "rel_l2".into());
    if !ok {
        out.insert("reason".into(), "rel_l2_above_threshold".into());
    }
    out
}

// ---------------------------------------------------------------------------- op-level goldens

/// One limit set of `KERNEL_THRESHOLDS`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelLimits {
    /// Limit of `rel_l2` over the finite elements.
    pub rel_l2: f64,
    /// Limit of the worst element divided by the largest golden element.
    pub max_rel: f64,
    /// Limit of `max_abs` when every finite golden element is zero.
    pub zero_atol: f64,
}

/// Written on 2026-11-03, BEFORE any CUDA output of these kernels had been recorded or compared. The reason is
/// arithmetic, not a measurement: float32 unit roundoff is 6e-8, and these kernels are short reductions with
/// an explicit FMA order, so 1e-5 leaves about 170 roundoffs for libm differences and a different lane split.
pub const KERNEL_THRESHOLDS: KernelLimits = KernelLimits {
    rel_l2: 1e-5,
    max_rel: 1e-5,
    zero_atol: 1e-6,
};

/// The smallest normal float32.
pub const FLOAT32_TINY: f64 = 1.1754943508222875e-38;

fn ulp_index(bits: u32) -> i64 {
    let magnitude = (bits & 0x7fff_ffff) as i64;
    if bits >> 31 == 1 {
        -magnitude
    } else {
        magnitude
    }
}

fn unravel(mut flat: usize, shape: &[usize]) -> Value {
    let mut index = vec![0usize; shape.len()];
    for (i, dim) in shape.iter().enumerate().rev() {
        if *dim > 0 {
            index[i] = flat % dim;
            flat /= dim;
        }
    }
    shape_value(&index)
}

fn first(mask: &[bool], shape: &[usize]) -> Value {
    unravel(mask.iter().position(|m| *m).unwrap_or(0), shape)
}

fn with(out: &Map<String, Value>, extra: &[(&str, Value)]) -> Map<String, Value> {
    let mut m = out.clone();
    for (k, v) in extra {
        m.insert((*k).into(), v.clone());
    }
    m
}

/// `compare_kernel_output(actual, expected, limits)`: one kernel output against its golden.
///
/// NaN and infinity must sit at the same elements with the same sign. Integer outputs must match exactly.
pub fn compare_kernel_output(
    actual: &Tensor,
    expected: &Tensor,
    limits: Option<KernelLimits>,
) -> Map<String, Value> {
    let limits = limits.unwrap_or(KERNEL_THRESHOLDS);
    let mut out = Map::new();
    out.insert("shape".into(), shape_value(&actual.shape));
    out.insert("expected_shape".into(), shape_value(&expected.shape));
    out.insert("dtype".into(), expected.numpy_dtype().into());
    for k in ["rel_l2", "max_rel", "max_abs", "max_ulp"] {
        out.insert(k.into(), Value::Null);
    }
    if actual.shape != expected.shape {
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "shape_mismatch".into()),
            ],
        );
    }
    if actual.numpy_dtype() != expected.numpy_dtype() {
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "dtype_mismatch".into()),
                ("actual_dtype", actual.numpy_dtype().into()),
            ],
        );
    }
    let shape = &expected.shape;
    if !expected.dtype.is_floating() {
        let (a, e) = (
            actual.to_i128().expect("integer"),
            expected.to_i128().expect("integer"),
        );
        let wrong: Vec<bool> = a.iter().zip(&e).map(|(x, y)| x != y).collect();
        let count = wrong.iter().filter(|w| **w).count();
        out.insert("mismatched".into(), count.into());
        if count > 0 {
            let at = wrong.iter().position(|w| *w).unwrap();
            return with(
                &out,
                &[
                    ("verdict", "fail".into()),
                    ("reason", "integer_mismatch".into()),
                    ("index", unravel(at, shape)),
                    ("golden", expected.item(at)),
                    ("port", actual.item(at)),
                ],
            );
        }
        return with(
            &out,
            &[("verdict", "pass".into()), ("basis", "exact".into())],
        );
    }
    let (a, e) = (actual.to_f64(), expected.to_f64());
    let a_nan: Vec<bool> = a.iter().map(|x| x.is_nan()).collect();
    let e_nan: Vec<bool> = e.iter().map(|x| x.is_nan()).collect();
    out.insert(
        "nan_in_golden".into(),
        e_nan.iter().filter(|x| **x).count().into(),
    );
    let nan_wrong: Vec<bool> = a_nan.iter().zip(&e_nan).map(|(x, y)| x != y).collect();
    if nan_wrong.iter().any(|w| *w) {
        let at = nan_wrong.iter().position(|w| *w).unwrap();
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "nan_mismatch".into()),
                (
                    "mismatched",
                    nan_wrong.iter().filter(|w| **w).count().into(),
                ),
                ("index", first(&nan_wrong, shape)),
                ("golden", num(e[at])),
                ("port", num(a[at])),
            ],
        );
    }
    let inf_wrong: Vec<bool> = a
        .iter()
        .zip(&e)
        .map(|(x, y)| (x.is_infinite() != y.is_infinite()) || (y.is_infinite() && x != y))
        .collect();
    if inf_wrong.iter().any(|w| *w) {
        let at = inf_wrong.iter().position(|w| *w).unwrap();
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "infinity_mismatch".into()),
                (
                    "mismatched",
                    inf_wrong.iter().filter(|w| **w).count().into(),
                ),
                ("index", first(&inf_wrong, shape)),
                ("golden", num(e[at])),
                ("port", num(a[at])),
            ],
        );
    }
    let finite: Vec<bool> = e.iter().map(|y| y.is_finite()).collect();
    out.insert(
        "bitwise_identical".into(),
        (actual.numpy_bytes() == expected.numpy_bytes()).into(),
    );
    let subnormal: Vec<bool> = e
        .iter()
        .zip(&finite)
        .map(|(y, f)| *f && y.abs() > 0.0 && y.abs() < FLOAT32_TINY)
        .collect();
    out.insert(
        "subnormal_in_golden".into(),
        subnormal.iter().filter(|s| **s).count().into(),
    );
    out.insert(
        "subnormal_flushed".into(),
        subnormal
            .iter()
            .zip(&a)
            .filter(|(s, x)| **s && **x == 0.0)
            .count()
            .into(),
    );
    if !finite.iter().any(|f| *f) {
        return with(
            &out,
            &[
                ("verdict", "pass".into()),
                ("basis", "all_special_values".into()),
            ],
        );
    }
    let diff: Vec<f64> = a
        .iter()
        .zip(&e)
        .zip(&finite)
        .map(|((x, y), f)| if *f { x - y } else { 0.0 })
        .collect();
    let worst_value = diff.iter().fold(0.0f64, |m, d| m.max(d.abs()));
    let worst = diff
        .iter()
        .position(|d| d.abs() == worst_value)
        .unwrap_or(0);
    let max_abs = diff[worst].abs();
    let finite_e: Vec<f64> = e
        .iter()
        .zip(&finite)
        .filter(|(_, f)| **f)
        .map(|(y, _)| *y)
        .collect();
    let scale = finite_e.iter().fold(0.0f64, |m, y| m.max(y.abs()));
    let golden_norm = norm(&finite_e);
    out.insert("max_abs".into(), num(max_abs));
    out.insert("index".into(), unravel(worst, shape));
    out.insert("golden".into(), num(e[worst]));
    out.insert("port".into(), num(a[worst]));
    if expected.numpy_dtype() == "float32" {
        let (ab, eb) = (
            actual.f32_bits().expect("float32"),
            expected.f32_bits().expect("float32"),
        );
        let ulp = ab
            .iter()
            .zip(&eb)
            .zip(&finite)
            .filter(|(_, f)| **f)
            .map(|((x, y), _)| (ulp_index(*x) - ulp_index(*y)).abs())
            .max()
            .unwrap_or(0);
        out.insert("max_ulp".into(), ulp.into());
    }
    if golden_norm == 0.0 {
        if max_abs <= limits.zero_atol {
            return with(
                &out,
                &[("verdict", "pass".into()), ("basis", "zero_atol".into())],
            );
        }
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "golden_is_zero_but_output_is_not".into()),
            ],
        );
    }
    let rel = norm(&diff) / golden_norm;
    let max_rel = max_abs / scale;
    out.insert("rel_l2".into(), num(rel));
    out.insert("max_rel".into(), num(max_rel));
    if rel > limits.rel_l2 {
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "rel_l2_above_limit".into()),
            ],
        );
    }
    if max_rel > limits.max_rel {
        return with(
            &out,
            &[
                ("verdict", "fail".into()),
                ("reason", "max_rel_above_limit".into()),
            ],
        );
    }
    with(
        &out,
        &[
            ("verdict", "pass".into()),
            ("basis", "rel_l2+max_rel".into()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(shape: &[usize], values: &[f64]) -> Array {
        Array {
            shape: shape.to_vec(),
            values: values.to_vec(),
        }
    }

    const F32: Limits = Limits {
        rel_l2: 1e-4,
        zero_atol: 1e-6,
    };

    #[test]
    fn identical_passes_and_one_percent_fails() {
        let x = arr(&[2, 2], &[1.0, -2.0, 3.0, 0.5]);
        assert_eq!(compare(&x, &x, F32)["verdict"], "pass");
        let y = arr(&[2, 2], &[1.01, -2.02, 3.03, 0.505]);
        let c = compare(&y, &x, F32);
        assert_eq!(c["reason"], "rel_l2_above_threshold");
        assert!((c["scale"].as_f64().unwrap() - 1.01).abs() < 1e-12);
    }

    #[test]
    fn shapes_reshape_or_fail() {
        let x = arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            compare(&arr(&[4], &[1.0, 2.0, 3.0, 4.0]), &x, F32)["reshaped"],
            true
        );
        assert_eq!(
            compare(&arr(&[3], &[1.0, 2.0, 3.0]), &x, F32)["reason"],
            "shape_mismatch"
        );
    }

    #[test]
    fn non_finite_and_zero_goldens() {
        let x = arr(&[2], &[1.0, 2.0]);
        let c = compare(&arr(&[2], &[f64::NAN, 2.0]), &x, F32);
        assert_eq!(
            (c["reason"].as_str(), c["nonfinite_count"].as_u64()),
            (Some("nonfinite_output"), Some(1))
        );
        assert_eq!(
            compare(&x, &arr(&[2], &[f64::INFINITY, 1.0]), F32)["reason"],
            "nonfinite_golden"
        );
        let z = arr(&[2], &[0.0, 0.0]);
        assert_eq!(
            compare(&arr(&[2], &[0.0, 1e-7]), &z, F32)["verdict"],
            "pass"
        );
        assert_eq!(
            compare(&arr(&[2], &[0.0, 1e-5]), &z, F32)["reason"],
            "golden_is_zero_but_output_is_not"
        );
    }

    #[test]
    fn thresholds_follow_python() {
        let t = resolve_threshold("bfloat16", None, false).unwrap();
        assert_eq!(
            (t.isolated_rel_l2, t.composite_rel_l2_max, t.noise_floor),
            (1.5e-2, 2e-1, 1e-4)
        );
        assert_eq!(noise_thresholds(&t, 1e-6), (1e-4, 1e-4));
        assert_eq!(noise_thresholds(&t, 1e-3), (6e-3, 0.1));
        assert_eq!(noise_thresholds(&t, 1.0), (1.5e-2, 2e-1));
        let p = resolve_threshold("float16", None, true).unwrap();
        assert_eq!(
            (p.isolated_rel_l2, p.composite_rel_l2_max, p.rel_l2),
            (1e-4, 1e-4, 5e-3)
        );
        let u = resolve_threshold("float32", Some(1e-3), false).unwrap();
        assert!(u.fixed_by_user && noise_thresholds(&u, 1.0) == (1e-3, 1e-3));
        assert!(resolve_threshold("int8", None, false).is_err());
        assert!(resolve_threshold("float32", Some(0.0), false).is_err());
    }

    #[test]
    fn kernel_outputs() {
        let g = Tensor::from_f32(vec![3], &[1.0, f32::NAN, f32::INFINITY]);
        assert_eq!(compare_kernel_output(&g, &g, None)["verdict"], "pass");
        let p = Tensor::from_f32(vec![3], &[1.0, 0.0, f32::INFINITY]);
        assert_eq!(
            compare_kernel_output(&p, &g, None)["reason"],
            "nan_mismatch"
        );
        let q = Tensor::from_f32(vec![3], &[1.0 + 1e-6, f32::NAN, f32::INFINITY]);
        let c = compare_kernel_output(&q, &g, None);
        assert_eq!(
            (c["verdict"].as_str(), c["max_ulp"].as_i64()),
            (Some("pass"), Some(8))
        );
    }
}
