//! Static weight comparison: the original checkpoint against a port, without running either model
//! (`src/cumlx/verify/weights.py`).
//!
//! This finds transposed, rescaled, sign-flipped, missing or wrongly shaped weights quickly, and it names the
//! parameter. It proves nothing about the port's code; `cumlx diff` does that.
//!
//! The Python command loads the port with mlx-lm. This one reads the port's `model*.safetensors` files as they
//! are stored, the same files that mlx-lm loads, so it needs no MLX. Three things follow, and each one ends in
//! a refusal or an `unverified` row, never in a pass:
//!
//! * A quantized port needs MLX to dequantize it: the command stops with `E_PYTHON_ONLY`.
//! * mlx-lm's `sanitize` step does not run. Point the command at a folder that mlx-lm wrote, where the names
//!   are already the port's names.
//! * The module type of a parameter is unknown, so a 3-D or 4-D weight whose shape is a channels-last
//!   convolution layout of the original is `unverified`, where Python compares it in that layout.

use crate::envelope::num;
use crate::errors::{CumlxError, CumlxResult};
use crate::metrics::{dot, norm, sum};
use crate::pyfmt::{format_g, py_path, repr_str};
use crate::safetensors::{is_truthy, Dtype, SafetensorsReader};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Schema of the weights report.
pub const WEIGHTS_SCHEMA: &str = "cumlx.weights-report.v1";
/// Unit roundoff: float16 4.9e-4, bfloat16 2.0e-3. float32 must be exact up to the source dtype.
pub const REL_L2_LIMITS: [(&str, f64); 3] =
    [("float32", 1e-6), ("float16", 1e-3), ("bfloat16", 5e-3)];
/// Measured on Qwen2.5-0.5B and SmolLM2-135M at 4 and 8 bits; see docs/CLI.md. Used by the Python command only,
/// because this one refuses a quantized port.
pub const QUANTIZED_MIN_COSINE: f64 = 0.9;
/// See [`QUANTIZED_MIN_COSINE`].
pub const QUANTIZED_MAX_SCALE_ERROR: f64 = 0.05;
/// A port with tied embeddings has no `lm_head.weight`; the original head must equal this parameter.
pub const TIED_EMBEDDING: &str = "model.embed_tokens.weight";

/// The `rel_l2` limit of a port dtype.
pub fn rel_l2_limit(kind: &str) -> f64 {
    REL_L2_LIMITS
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, v)| *v)
        .unwrap_or(REL_L2_LIMITS[0].1)
}

fn expert_proj(name: &str) -> Option<&'static str> {
    Some(match name {
        "w1" | "gate_proj" => "gate_proj",
        "w2" | "down_proj" => "down_proj",
        "w3" | "up_proj" => "up_proj",
        _ => return None,
    })
}

/// A float64 array with a shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Nd {
    /// Shape.
    pub shape: Vec<usize>,
    /// Values in C order.
    pub values: Vec<f64>,
}

impl Nd {
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The transpose of a 2-D array.
    fn t(&self) -> Nd {
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut values = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                values[j * r + i] = self.values[i * c + j];
            }
        }
        Nd {
            shape: vec![c, r],
            values,
        }
    }

    /// Item `i` along the first axis.
    fn index(&self, i: usize) -> Nd {
        let inner: usize = self.shape[1..].iter().product();
        Nd {
            shape: self.shape[1..].to_vec(),
            values: self.values[i * inner..(i + 1) * inner].to_vec(),
        }
    }

    /// `np.split(a, 2, axis=0)`.
    fn halves(&self) -> Option<(Nd, Nd)> {
        if self.shape.is_empty() || self.shape[0] % 2 != 0 {
            return None;
        }
        let half = self.shape[0] / 2;
        let inner: usize = self.shape[1..].iter().product();
        let mut shape = self.shape.clone();
        shape[0] = half;
        let (a, b) = self.values.split_at(half * inner);
        Some((
            Nd {
                shape: shape.clone(),
                values: a.to_vec(),
            },
            Nd {
                shape,
                values: b.to_vec(),
            },
        ))
    }

    /// The channels-last convolution layout: axes `(0, 2, 1)` or `(0, 3, 1, 2)` as NumPy transposes them.
    fn conv_layout_shape(&self) -> Vec<usize> {
        match self.shape.as_slice() {
            [a, b, c] => vec![*a, *c, *b],
            [a, b, c, d] => vec![*a, *d, *b, *c],
            other => other.to_vec(),
        }
    }
}

fn shape_value(shape: &[usize]) -> Value {
    shape.iter().map(|s| Value::from(*s)).collect()
}

fn close(a: &Nd, b: &Nd) -> bool {
    if a.shape != b.shape {
        return false;
    }
    let d: Vec<f64> = a.values.iter().zip(&b.values).map(|(x, y)| x - y).collect();
    let nb = norm(&b.values);
    norm(&d) <= 1e-2 * if 1e-30 > nb { 1e-30 } else { nb }
}

/// `_diagnose`: name the defect when the two arrays are an exact simple transform of each other.
pub fn diagnose(port: &Nd, orig: &Nd) -> Option<String> {
    if port.ndim() == 2 && close(&port.t(), orig) {
        return Some("transposed".into());
    }
    let negated = Nd {
        shape: port.shape.clone(),
        values: port.values.iter().map(|x| -x).collect(),
    };
    if close(&negated, orig) {
        return Some("sign_flipped".into());
    }
    if port.shape != orig.shape {
        return None; // NumPy would broadcast or raise here; neither names a scale.
    }
    let d = sum(orig.values.iter().map(|x| x * x));
    if d > 0.0 {
        let scale = sum(port.values.iter().zip(&orig.values).map(|(p, o)| p * o)) / d;
        let scaled = Nd {
            shape: orig.shape.clone(),
            values: orig.values.iter().map(|o| scale * o).collect(),
        };
        if (scale - 1.0).abs() > 1e-2 && close(port, &scaled) {
            return Some(format!("scaled_by_{}", format_g(scale, 6)));
        }
    }
    None
}

fn diagnose_same(a: &Nd, b: &Nd) -> bool {
    if a.shape != b.shape {
        return false;
    }
    let d: Vec<f64> = a.values.iter().zip(&b.values).map(|(x, y)| x - y).collect();
    norm(&d) <= 0.5 * norm(&b.values)
}

/// `_other_expert`: the expert index whose port weights hold this original expert, when it is not its own.
fn other_expert(port: &Nd, orig: &Nd, expert: usize) -> Option<usize> {
    (0..port.shape.first().copied().unwrap_or(0))
        .find(|e| *e != expert && diagnose_same(&port.index(*e), orig))
}

/// `_compare`: one parameter against its original. The verdict is under `verdict`.
pub fn compare_parameter(port: &Nd, orig: &Nd, kind: &str) -> Map<String, Value> {
    let mut out = Map::new();
    if port.shape != orig.shape {
        out.insert("verdict".into(), "fail".into());
        out.insert("reason".into(), "shape_mismatch".into());
        out.insert("port_shape".into(), shape_value(&port.shape));
        out.insert("original_shape".into(), shape_value(&orig.shape));
        if port.ndim() == 2 && port.t().shape == orig.shape {
            let d = if diagnose(port, orig).as_deref() == Some("transposed") {
                "transposed"
            } else {
                "shape_is_transposed"
            };
            out.insert("diagnosis".into(), d.into());
        }
        return out;
    }
    let (o, p) = (&orig.values, &port.values);
    if !p.iter().all(|x| x.is_finite()) {
        out.insert("verdict".into(), "fail".into());
        out.insert("reason".into(), "nonfinite_port_weight".into());
        return out;
    }
    let diff: Vec<f64> = p.iter().zip(o).map(|(x, y)| x - y).collect();
    let (on, pn, dn) = (norm(o), norm(p), norm(&diff));
    let rel = if on > 0.0 {
        dn / on
    } else if dn == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let po = dot(p, o);
    let cosine = if pn > 0.0 && on > 0.0 {
        po / (pn * on)
    } else if pn == on {
        1.0
    } else {
        0.0
    };
    let scale = if on > 0.0 { po / (on * on) } else { 1.0 };
    out.insert("rel_l2".into(), num(rel));
    out.insert("cosine".into(), num(cosine));
    out.insert("scale".into(), num(scale));
    let ok = if kind == "quantized" {
        cosine >= QUANTIZED_MIN_COSINE && (scale - 1.0).abs() <= QUANTIZED_MAX_SCALE_ERROR
    } else {
        rel <= rel_l2_limit(kind)
    };
    out.insert("verdict".into(), if ok { "pass" } else { "fail" }.into());
    if !ok {
        let reason = if kind == "quantized" {
            "quantized_direction_or_scale"
        } else {
            "rel_l2_above_limit"
        };
        out.insert("reason".into(), reason.into());
        if let Some(d) = diagnose(port, orig) {
            out.insert("diagnosis".into(), d.into());
        }
    }
    out
}

fn looks_local(path: &str) -> bool {
    path.starts_with('/') || path.starts_with('.') || path.starts_with('~')
}

/// The `E_PYTHON_ONLY` error of an option or input that only the Python cumlx handles.
pub fn python_only(message: impl Into<String>, python_command: &str) -> CumlxError {
    CumlxError::new("E_PYTHON_ONLY", message).hint(format!(
        "python -m pip install 'cumlx-lab[diff]' && {python_command}"
    ))
}

fn sorted_glob(dir: &Path, accept: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_str().is_some_and(&accept))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// `checkpoint_files`: the safetensors files of the original checkpoint, from a local file or folder.
pub fn checkpoint_files(original: &str, python_command: &str) -> CumlxResult<Vec<PathBuf>> {
    let root = Path::new(original);
    if !root.exists() {
        if looks_local(original) {
            return Err(CumlxError::new(
                "E_INPUT_NOT_FOUND",
                format!("checkpoint not found: {}", py_path(original)),
            ));
        }
        return Err(python_only(
            format!(
                "{} is not a local file or folder; cumlx-rs reads local checkpoints only, and a Hugging Face id needs the Python cumlx",
                repr_str(original)
            ),
            python_command,
        ));
    }
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        let shown = py_path(&index.to_string_lossy());
        let bad = |why: String| {
            CumlxError::new(
                "E_INPUT_INVALID",
                format!("{shown} is not a safetensors index: {why}"),
            )
        };
        let text = std::fs::read(&index).map_err(|e| bad(e.to_string()))?;
        let doc: Value = serde_json::from_slice(&text).map_err(|e| bad(e.to_string()))?;
        let map = doc
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("no weight_map".into()))?;
        let mut names = BTreeSet::new();
        for v in map.values() {
            names.insert(
                v.as_str()
                    .ok_or_else(|| bad("a file name is not a string".into()))?
                    .to_string(),
            );
        }
        return Ok(names.into_iter().map(|n| root.join(n)).collect());
    }
    let files = sorted_glob(root, |n| n.ends_with(".safetensors"));
    if files.is_empty() {
        return Err(CumlxError::new(
            "E_INPUT_NOT_FOUND",
            format!("no *.safetensors in {}", py_path(original)),
        )
        .hint("only safetensors checkpoints can be read without torch"));
    }
    Ok(files)
}

/// The port's floating parameters, read from the files that mlx-lm loads.
pub struct Port {
    readers: Vec<SafetensorsReader>,
    params: BTreeMap<String, usize>,
}

impl Port {
    /// Open a port folder in mlx-lm format: `config.json` and `model*.safetensors`.
    pub fn open(port: &str, python_command: &str) -> CumlxResult<Port> {
        let p = Path::new(port);
        if !p.exists() {
            if looks_local(port) {
                return Err(CumlxError::new(
                    "E_INPUT_NOT_FOUND",
                    format!("MLX port directory not found: {port}"),
                )
                .hint(
                    "pass an existing directory, or a Hugging Face id such as mlx-community/<name>",
                ));
            }
            return Err(python_only(
                format!("{} is not a local folder; cumlx-rs reads local ports only, and a Hugging Face id needs the Python cumlx", repr_str(port)),
                python_command,
            ));
        }
        let load_error = |why: &str| {
            CumlxError::new(
                "E_MODEL_LOAD",
                format!("cumlx-rs could not read the port {}: {why}", repr_str(port)),
            )
            .hint("pass the folder that mlx-lm wrote: config.json and model*.safetensors")
        };
        if !p.is_dir() {
            return Err(load_error("it is not a folder"));
        }
        let config_path = p.join("config.json");
        let config: Value = std::fs::read(&config_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .ok_or_else(|| load_error("config.json is missing or is not JSON"))?;
        let text_config = config.get("text_config");
        // The keys that make mlx-lm quantize the model while it loads it (`mlx_lm.utils.load_model`).
        let quantized = config.get("quantization").is_some_and(|v| !v.is_null())
            || config.get("quantization_config").is_some_and(is_truthy)
            || text_config
                .and_then(|t| t.get("quantization_config"))
                .is_some_and(is_truthy)
            || p.join("hf_quant_config.json").exists();
        let refuse_quantized = || {
            python_only(
                format!(
                    "the port {} is quantized, and dequantizing it needs MLX",
                    repr_str(port)
                ),
                python_command,
            )
        };
        if quantized {
            return Err(refuse_quantized());
        }
        let files = sorted_glob(p, |n| n.starts_with("model") && n.ends_with(".safetensors"));
        if files.is_empty() {
            return Err(load_error("it holds no model*.safetensors"));
        }
        let mut readers = Vec::new();
        let mut params = BTreeMap::new();
        for f in files {
            let r = SafetensorsReader::open(&f)?;
            for name in r.keys() {
                if name.ends_with(".scales") {
                    return Err(refuse_quantized());
                }
                if r.dtype(name).is_some_and(Dtype::is_floating) {
                    params.insert(name.to_string(), readers.len());
                }
            }
            readers.push(r);
        }
        Ok(Port { readers, params })
    }

    /// True when the port has the floating parameter `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Names of every floating parameter, sorted.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.params.keys()
    }

    /// The dtype name of a parameter, as Python's `kinds` holds it.
    pub fn kind(&self, name: &str) -> &'static str {
        match self
            .params
            .get(name)
            .and_then(|i| self.readers[*i].dtype(name))
        {
            Some(Dtype::F64) => "float64",
            Some(Dtype::F16) => "float16",
            Some(Dtype::BF16) => "bfloat16",
            _ => "float32",
        }
    }

    /// A parameter as float64. Python reads every port parameter as float32 (`astype(mx.float32)`), so a float64
    /// parameter is rounded to float32 first; the other float types widen exactly.
    pub fn get(&self, name: &str) -> CumlxResult<Nd> {
        let i = self.params[name];
        let t = self.readers[i].get(name)?;
        let values = t.to_f64().into_iter().map(|v| v as f32 as f64).collect();
        Ok(Nd {
            shape: t.shape,
            values,
        })
    }
}

struct ExpertPart {
    row_name: String,
    key: String,
    expert: usize,
    array: Nd,
}

fn split_single(name: &str) -> Option<(&str, usize, &str)> {
    let stem = name.strip_suffix(".weight")?;
    let (head, proj) = stem.rsplit_once('.')?;
    if proj.is_empty() || !proj.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let (head, index) = head.rsplit_once('.')?;
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let block = head.strip_suffix(".experts")?;
    if block.is_empty() {
        return None;
    }
    Some((block, index.parse().ok()?, proj))
}

/// `_expert_parts`: original MoE expert tensors as (row name, stacked port name, expert index, original array).
fn expert_parts(name: &str, orig: &Nd, port: &Port) -> CumlxResult<Option<Vec<ExpertPart>>> {
    let stacked = |block: &str, proj: &str| -> Option<String> {
        for candidate in [
            block.to_string(),
            block.replace(".mlp", ".block_sparse_moe"),
        ] {
            let key = format!("{candidate}.switch_mlp.{proj}.weight");
            if port.contains(&key) {
                return Some(key);
            }
        }
        None
    };
    if let Some((block, expert, proj)) = split_single(name) {
        if let Some(target) = expert_proj(proj) {
            let key = stacked(&block.replace(".block_sparse_moe", ".mlp"), target);
            return Ok(key.map(|key| {
                vec![ExpertPart {
                    row_name: name.into(),
                    key,
                    expert,
                    array: orig.clone(),
                }]
            }));
        }
    }
    let fused = ["gate_up_proj", "down_proj"].into_iter().find_map(|tail| {
        name.strip_suffix(&format!(".experts.{tail}"))
            .filter(|b| !b.is_empty())
            .map(|b| (b, tail))
    });
    let Some((block, tail)) = fused else {
        return Ok(None);
    };
    if orig.shape.is_empty() {
        return Err(CumlxError::new(
            "E_INPUT_INVALID",
            format!("{} is a scalar, not a stack of experts", repr_str(name)),
        ));
    }
    let mut rows = Vec::new();
    for e in 0..orig.shape[0] {
        let item = orig.index(e);
        let parts = if tail == "down_proj" {
            vec![("down_proj", item)]
        } else {
            let (gate, up) = item.halves().ok_or_else(|| {
                CumlxError::new(
                    "E_INPUT_INVALID",
                    format!("{} does not split into gate and up halves", repr_str(name)),
                )
            })?;
            vec![("gate_proj", gate), ("up_proj", up)]
        };
        for (proj, array) in parts {
            let Some(key) = stacked(block, proj) else {
                return Ok(None);
            };
            rows.push(ExpertPart {
                row_name: format!("{block}.experts.{e}.{proj}.weight"),
                key,
                expert: e,
                array,
            });
        }
    }
    Ok(Some(rows))
}

fn row(name: &str, port_name: &str, kind: Option<&str>, status: &str) -> Map<String, Value> {
    let mut r = Map::new();
    r.insert("name".into(), name.into());
    r.insert("port_name".into(), port_name.into());
    if let Some(k) = kind {
        r.insert("kind".into(), k.into());
    }
    r.insert("status".into(), status.into());
    r
}

/// `run_weights(original, port)`: the report that Python returns, for an unquantized port read from its files.
pub fn run_weights(
    original: &str,
    port_path: &str,
    python_command: &str,
) -> CumlxResult<Map<String, Value>> {
    let mut readers: Vec<SafetensorsReader> = Vec::new();
    let mut owner: BTreeMap<String, usize> = BTreeMap::new();
    for f in checkpoint_files(original, python_command)? {
        let r = SafetensorsReader::open(&f)?;
        for name in r.keys() {
            owner.insert(name.to_string(), readers.len());
        }
        readers.push(r);
    }
    if owner.is_empty() {
        return Err(CumlxError::new(
            "E_INPUT_INVALID",
            format!("no tensors in the checkpoint {}", repr_str(original)),
        ));
    }
    let port = Port::open(port_path, python_command)?;
    let mut rows: Vec<Map<String, Value>> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (name, idx) in &owner {
        let reader = &readers[*idx];
        let mut target = name.clone();
        let tensor = reader.get(name)?;
        let float = tensor.dtype.is_floating();
        let orig = Nd {
            shape: tensor.shape.clone(),
            values: tensor.to_f64(),
        };
        let parts = if float && !port.contains(&target) {
            expert_parts(&target, &orig, &port)?
        } else {
            None
        };
        if let Some(parts) = parts.filter(|p| !p.is_empty()) {
            for part in parts {
                seen.insert(part.key.clone());
                let stack = port.get(&part.key)?;
                if part.expert >= stack.shape.first().copied().unwrap_or(0) {
                    return Err(CumlxError::new(
                        "E_INPUT_INVALID",
                        format!(
                            "the port's {} has no expert {}",
                            repr_str(&part.key),
                            part.expert
                        ),
                    ));
                }
                let kind = port.kind(&part.key);
                let mut c = compare_parameter(&stack.index(part.expert), &part.array, kind);
                if c["verdict"] == "fail" {
                    if let Some(other) = other_expert(&stack, &part.array, part.expert) {
                        c.insert(
                            "diagnosis".into(),
                            format!("expert_order: this expert is at index {other} in the port")
                                .into(),
                        );
                    }
                }
                let verdict = c.shift_remove("verdict").expect("verdict");
                let mut r = row(
                    &part.row_name,
                    &format!("{}[{}]", part.key, part.expert),
                    Some(kind),
                    verdict.as_str().unwrap(),
                );
                r.extend(c);
                rows.push(r);
            }
            continue;
        }
        if !port.contains(&target) && target == "lm_head.weight" && port.contains(TIED_EMBEDDING) {
            target = TIED_EMBEDDING.into(); // the port shares the embedding, so the original head must equal it
        }
        if !port.contains(&target) {
            let mut r = row(name, &target, None, "unverified");
            r.insert("reason".into(), "no such parameter in the port".into());
            rows.push(r);
            continue;
        }
        seen.insert(target.clone());
        if !float {
            let mut r = row(name, &target, None, "ignored");
            r.insert(
                "reason".into(),
                format!("non-float {}", tensor.numpy_dtype()).into(),
            );
            rows.push(r);
            continue;
        }
        let port_array = port.get(&target)?;
        let kind = port.kind(&target);
        if port_array.shape != orig.shape
            && matches!(port_array.ndim(), 3 | 4)
            && port_array.conv_layout_shape() == orig.shape
        {
            let mut r = row(name, &target, Some(kind), "unverified");
            r.insert(
                "reason".into(),
                "the port shape is a channels-last convolution layout of the original; cumlx-rs cannot see the module type, run the Python cumlx weights".into(),
            );
            rows.push(r);
            continue;
        }
        let mut c = compare_parameter(&port_array, &orig, kind);
        let verdict = c.shift_remove("verdict").expect("verdict");
        let mut r = row(name, &target, Some(kind), verdict.as_str().unwrap());
        r.extend(c);
        rows.push(r);
    }
    let extra: Vec<Value> = port
        .names()
        .filter(|n| !seen.contains(*n))
        .map(|n| Value::from(n.as_str()))
        .collect();
    let count = |k: &str| rows.iter().filter(|r| r["status"] == k).count();
    let (pass, fail, unverified, ignored) = (
        count("pass"),
        count("fail"),
        count("unverified"),
        count("ignored"),
    );
    let mut verdict = if fail > 0 {
        "diverged"
    } else if unverified > 0 {
        "incomplete"
    } else {
        "ok"
    };
    if pass == 0 && verdict == "ok" {
        verdict = "incomplete";
    }
    let worst = rows
        .iter()
        .find(|r| r["status"] == "fail")
        .map(|r| Value::Object(r.clone()))
        .unwrap_or(Value::Null);
    let mut summary = Map::new();
    summary.insert("pass".into(), pass.into());
    summary.insert("fail".into(), fail.into());
    summary.insert("unverified".into(), unverified.into());
    summary.insert("ignored".into(), ignored.into());
    summary.insert("parameters".into(), rows.len().into());
    summary.insert("extra_in_port".into(), extra.len().into());
    let mut report = Map::new();
    report.insert("schema".into(), WEIGHTS_SCHEMA.into());
    report.insert("verdict".into(), verdict.into());
    report.insert("original".into(), original.into());
    report.insert("port".into(), port_path.into());
    report.insert("summary".into(), Value::Object(summary));
    report.insert("first_failure".into(), worst);
    report.insert("extra_in_port".into(), Value::Array(extra));
    report.insert(
        "parameters".into(),
        Value::Array(rows.into_iter().map(Value::Object).collect()),
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nd(shape: &[usize], values: &[f64]) -> Nd {
        Nd {
            shape: shape.to_vec(),
            values: values.to_vec(),
        }
    }

    #[test]
    fn diagnoses_name_the_transform() {
        let o = nd(&[2, 2], &[1.0, 2.0, -3.0, 4.5]);
        assert_eq!(diagnose(&o.t(), &o).as_deref(), Some("transposed"));
        assert_eq!(
            diagnose(&nd(&[2, 2], &[-1.0, -2.0, 3.0, -4.5]), &o).as_deref(),
            Some("sign_flipped")
        );
        assert_eq!(
            diagnose(&nd(&[2, 2], &[2.0, 4.0, -6.0, 9.0]), &o).as_deref(),
            Some("scaled_by_2")
        );
        assert_eq!(
            diagnose(&nd(&[2, 2], &[1.5, 3.0, -4.5, 6.75]), &o).as_deref(),
            Some("scaled_by_1.5")
        );
        assert_eq!(diagnose(&nd(&[2, 2], &[1.0, 0.0, 0.0, 1.0]), &o), None);
    }

    #[test]
    fn compare_parameter_follows_python() {
        let o = nd(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = compare_parameter(&o, &o, "float32");
        assert_eq!(
            (c["verdict"].as_str(), c["rel_l2"].as_f64()),
            (Some("pass"), Some(0.0))
        );
        let t = compare_parameter(&o.t(), &o, "float32");
        assert_eq!(
            (t["reason"].as_str(), t["diagnosis"].as_str()),
            (Some("shape_mismatch"), Some("transposed"))
        );
        let mut v = o.clone();
        v.values[0] = f64::NAN;
        assert_eq!(
            compare_parameter(&v, &o, "float32")["reason"],
            "nonfinite_port_weight"
        );
        let half = nd(&[2, 3], &[1.0005, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(compare_parameter(&half, &o, "float16")["verdict"], "pass");
        assert_eq!(compare_parameter(&half, &o, "float32")["verdict"], "fail");
    }

    #[test]
    fn expert_names_split() {
        assert_eq!(
            split_single("model.layers.0.mlp.experts.3.w1.weight"),
            Some(("model.layers.0.mlp", 3, "w1"))
        );
        assert_eq!(split_single("model.layers.0.mlp.experts.x.w1.weight"), None);
        assert_eq!(split_single("model.layers.0.mlp.gate.weight"), None);
    }
}
