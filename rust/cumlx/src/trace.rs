//! Reading and checking a golden trace `cumlx.trace.v1` (`docs/TRACE-FORMAT.md`).
//!
//! The checks follow `src/cumlx/verify/trace.py`, which is stricter than the format document in one place:
//! a sample, record or tensor reference with a field that the Python dataclass does not know is rejected,
//! although the document says that a reader must ignore unknown fields. This reader does the same as the
//! Python reader, so that both give the same verdict on the same trace.

use crate::errors::{CumlxError, CumlxResult};
use crate::pyfmt::{py_path, py_path_join, py_repr};
use crate::safetensors::{is_truthy, SafetensorsReader, Tensor};
use crate::sha256::file_sha256;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// Schema name of a trace manifest.
pub const TRACE_SCHEMA: &str = "cumlx.trace.v1";
/// File name of the manifest.
pub const MANIFEST: &str = "manifest.json";
/// Accepted values of `golden.kind`.
pub const GOLDEN_KINDS: [&str; 2] = ["reference", "quantization-aware"];

/// One recorded tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    /// Name of the tensor in the sample file.
    pub tensor: Value,
    /// Shape as recorded.
    pub shape: Value,
    /// Dtype as recorded.
    pub dtype: Value,
    /// Where it came from: `arg0`, `kw:hidden_states`, `tensor`, `tuple[0]`, `ModelOutput[0]`.
    pub source: Value,
}

/// One call of one module.
#[derive(Clone, Debug, PartialEq)]
pub struct ModuleRecord {
    /// `<module name>#<call index>`.
    pub key: Value,
    /// Module name; the root is `""`.
    pub name: Value,
    /// Call index of this module in the sample.
    pub call: Value,
    /// Completion order.
    pub order: Value,
    /// Class name of the module.
    pub r#type: Value,
    /// True when the module owns parameters.
    pub has_params: Value,
    /// True when a reader can replay the module on its golden input.
    pub replayable: Value,
    /// Free text.
    pub replay_note: Value,
    /// Recorded inputs.
    pub inputs: Vec<TensorRef>,
    /// Recorded output, if any.
    pub output: Option<TensorRef>,
    /// `single`, `hidden`, `hidden_cached`, `tokens_cached`, `experts` or null.
    pub replay_kind: Value,
    /// `prefill` or `decode`.
    pub phase: Value,
}

/// One input of the model.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// `s0000`, ...
    pub id: Value,
    /// Path of the sample file, relative to the trace.
    pub file: Value,
    /// SHA-256 of the sample file.
    pub sha256: Value,
    /// Number of prefill tokens.
    pub num_tokens: Value,
    /// Records in completion order.
    pub records: Vec<ModuleRecord>,
    /// Greedy decode steps recorded on the KV cache.
    pub decode_steps: Value,
}

/// `manifest.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    /// What produced the golden.
    pub source: Value,
    /// `{kind, dtype, ...}`.
    pub golden: Map<String, Value>,
    /// One entry per input.
    pub samples: Vec<Sample>,
    /// Always [`TRACE_SCHEMA`].
    pub schema: String,
    /// Version of the writer.
    pub cumlx_version: Value,
    /// Creation time.
    pub created_utc: Value,
}

fn malformed(what: String) -> CumlxError {
    CumlxError::new(
        "E_INPUT_INVALID",
        format!("trace manifest is malformed: {what}"),
    )
}

/// The fields of a Python dataclass constructed with `**d`: every required one present, no unknown one.
fn fields<'a>(
    d: &'a Value,
    class: &str,
    required: &[&str],
    optional: &[&str],
) -> CumlxResult<&'a Map<String, Value>> {
    let Some(map) = d.as_object() else {
        return Err(malformed(format!(
            "{class} must be an object, not {}",
            py_repr(d)
        )));
    };
    if let Some(unknown) = map
        .keys()
        .find(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
    {
        return Err(malformed(format!(
            "{class} has the unknown field {}",
            crate::pyfmt::repr_str(unknown)
        )));
    }
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|k| !map.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        return Err(malformed(format!(
            "{class} lacks the field(s) {}",
            missing
                .iter()
                .map(|m| crate::pyfmt::repr_str(m))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(map)
}

fn get(map: &Map<String, Value>, key: &str) -> Value {
    map.get(key).cloned().unwrap_or(Value::Null)
}

impl TensorRef {
    fn from_value(d: &Value) -> CumlxResult<TensorRef> {
        let m = fields(d, "TensorRef", &["tensor", "shape", "dtype", "source"], &[])?;
        Ok(TensorRef {
            tensor: get(m, "tensor"),
            shape: get(m, "shape"),
            dtype: get(m, "dtype"),
            source: get(m, "source"),
        })
    }
}

impl ModuleRecord {
    fn from_value(d: &Value) -> CumlxResult<ModuleRecord> {
        let m = fields(
            d,
            "ModuleRecord",
            &[
                "key",
                "name",
                "call",
                "order",
                "type",
                "has_params",
                "replayable",
            ],
            &["replay_note", "inputs", "output", "replay_kind", "phase"],
        )?;
        let inputs = match m.get("inputs") {
            None => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(TensorRef::from_value)
                .collect::<CumlxResult<_>>()?,
            Some(other) => {
                return Err(malformed(format!(
                    "record inputs must be a list, not {}",
                    py_repr(other)
                )))
            }
        };
        let output = match m.get("output") {
            Some(v) if is_truthy(v) => Some(TensorRef::from_value(v)?),
            _ => None,
        };
        let mut replay_kind = get(m, "replay_kind");
        if replay_kind.is_null() && m.get("replayable").is_some_and(is_truthy) {
            replay_kind = "single".into(); // traces written before replay_kind existed
        }
        Ok(ModuleRecord {
            key: get(m, "key"),
            name: get(m, "name"),
            call: get(m, "call"),
            order: get(m, "order"),
            r#type: get(m, "type"),
            has_params: get(m, "has_params"),
            replayable: get(m, "replayable"),
            replay_note: get(m, "replay_note"),
            inputs,
            output,
            replay_kind,
            phase: m.get("phase").cloned().unwrap_or_else(|| "prefill".into()),
        })
    }

    /// The record key as a string for lookups (`r.key` is a string in every trace that CUMLX writes).
    pub fn key_text(&self) -> String {
        match &self.key {
            Value::String(s) => s.clone(),
            other => py_repr(other),
        }
    }
}

impl Sample {
    fn from_value(d: &Value) -> CumlxResult<Sample> {
        let m = fields(
            d,
            "Sample",
            &["id", "file", "sha256", "num_tokens", "records"],
            &["decode_steps"],
        )?;
        let records = match &m["records"] {
            Value::Array(items) => items
                .iter()
                .map(ModuleRecord::from_value)
                .collect::<CumlxResult<_>>()?,
            other => {
                return Err(malformed(format!(
                    "sample records must be a list, not {}",
                    py_repr(other)
                )))
            }
        };
        Ok(Sample {
            id: get(m, "id"),
            file: get(m, "file"),
            sha256: get(m, "sha256"),
            num_tokens: get(m, "num_tokens"),
            records,
            decode_steps: m.get("decode_steps").cloned().unwrap_or_else(|| 0.into()),
        })
    }

    /// The sample id as text (`s0000`).
    pub fn id_text(&self) -> String {
        crate::pyfmt::py_str(&self.id)
    }
}

impl Manifest {
    /// Check and read a parsed `manifest.json`.
    pub fn from_value(d: &Value) -> CumlxResult<Manifest> {
        let schema = d.as_object().and_then(|m| m.get("schema"));
        if schema.and_then(Value::as_str) != Some(TRACE_SCHEMA) {
            let found = if d.is_object() {
                schema.cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            return Err(CumlxError::new(
                "E_INPUT_INVALID",
                format!("trace manifest schema is not {TRACE_SCHEMA}"),
            )
            .hint("re-capture the trace with this CUMLX version")
            .detail("found", found));
        }
        let m = d.as_object().expect("checked above");
        for key in ["source", "golden", "samples"] {
            if !m.contains_key(key) {
                return Err(malformed(crate::pyfmt::repr_str(key)));
            }
        }
        let samples = match &m["samples"] {
            Value::Array(items) => items
                .iter()
                .map(Sample::from_value)
                .collect::<CumlxResult<Vec<_>>>()?,
            other => {
                return Err(malformed(format!(
                    "samples must be a list, not {}",
                    py_repr(other)
                )))
            }
        };
        let Value::Object(golden) = &m["golden"] else {
            return Err(CumlxError::new(
                "E_INPUT_INVALID",
                "trace manifest has an unknown golden.kind",
            ));
        };
        if !golden
            .get("kind")
            .and_then(Value::as_str)
            .is_some_and(|k| GOLDEN_KINDS.contains(&k))
        {
            return Err(CumlxError::new(
                "E_INPUT_INVALID",
                "trace manifest has an unknown golden.kind",
            ));
        }
        Ok(Manifest {
            source: m["source"].clone(),
            golden: golden.clone(),
            samples,
            schema: TRACE_SCHEMA.into(),
            cumlx_version: m.get("cumlx_version").cloned().unwrap_or_else(|| "".into()),
            created_utc: m.get("created_utc").cloned().unwrap_or_else(|| "".into()),
        })
    }
}

/// An opened trace: the manifest and one reader per sample.
#[derive(Debug)]
pub struct Trace {
    /// The trace directory.
    pub dir: PathBuf,
    /// The checked manifest.
    pub manifest: Manifest,
    readers: Vec<SafetensorsReader>,
}

impl Trace {
    /// Open a trace. With `verify`, every sample file must match its SHA-256.
    pub fn open(path: &Path, verify: bool) -> CumlxResult<Trace> {
        let dir_text = path.to_string_lossy().to_string();
        let shown = py_path(&dir_text);
        let mpath = path.join(MANIFEST);
        let mshown = py_path_join(&dir_text, MANIFEST);
        if !mpath.is_file() {
            return Err(
                CumlxError::new("E_INPUT_NOT_FOUND", format!("no {MANIFEST} in {shown}"))
                    .hint("pass the directory written by `cumlx capture -o DIR`"),
            );
        }
        let bytes = std::fs::read(&mpath).map_err(|e| {
            CumlxError::new(
                "E_INPUT_INVALID",
                format!("{mshown} could not be read: {e}"),
            )
        })?;
        let raw: Value = serde_json::from_slice(&bytes).map_err(|e| {
            CumlxError::new(
                "E_INPUT_INVALID",
                format!("{mshown} is not valid JSON: {e}"),
            )
        })?;
        let manifest = Manifest::from_value(&raw)?;
        let mut readers = Vec::with_capacity(manifest.samples.len());
        for s in &manifest.samples {
            let Value::String(file) = &s.file else {
                return Err(malformed(format!(
                    "sample file must be a string, not {}",
                    py_repr(&s.file)
                )));
            };
            let p = path.join(file);
            let pshown = py_path_join(&dir_text, file);
            if !p.is_file() {
                return Err(CumlxError::new(
                    "E_INPUT_NOT_FOUND",
                    format!("trace sample file missing: {pshown}"),
                ));
            }
            if verify {
                let digest = file_sha256(&p).map_err(|e| {
                    CumlxError::new(
                        "E_INPUT_INVALID",
                        format!("{pshown} could not be read: {e}"),
                    )
                })?;
                if s.sha256.as_str() != Some(digest.as_str()) {
                    return Err(CumlxError::new(
                        "E_INPUT_INVALID",
                        format!(
                            "checksum mismatch for {pshown}; the trace was modified or truncated"
                        ),
                    ));
                }
            }
            readers.push(SafetensorsReader::open(&p)?);
        }
        Ok(Trace {
            dir: path.to_path_buf(),
            manifest,
            readers,
        })
    }

    /// Read one tensor of sample `index`.
    pub fn tensor(&self, index: usize, name: &Value) -> CumlxResult<Tensor> {
        match name {
            Value::String(s) => self.readers[index].get(s),
            other => self.readers[index].get(&py_repr(other)),
        }
    }

    /// The prefill token ids of sample `index`.
    pub fn token_ids(&self, index: usize) -> CumlxResult<Tensor> {
        self.readers[index].get("token_ids")
    }

    /// The reader of sample `index`.
    pub fn reader(&self, index: usize) -> &SafetensorsReader {
        &self.readers[index]
    }
}
