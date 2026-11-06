//! A safetensors reader with the same checks and errors as `src/cumlx/verify/safetensors_io.py`.
//!
//! Format: an 8-byte little-endian header length, a JSON header, then one byte buffer. The header maps tensor
//! names to dtype, shape and `[begin, end)` offsets into that buffer, plus an optional `__metadata__` map.
//!
//! The reader is written here, and not taken from the `safetensors` crate, so that it accepts and refuses the
//! same files as the Python reader: for example the FP4 scale types `F8_E4M3` and `F8_E8M0`, which it reads as
//! raw `uint8` codes, and a buffer with unused bytes between tensors.

use crate::errors::{CumlxError, CumlxResult};
use crate::pyfmt::{py_path, repr_str};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// The largest header the reader accepts, as in Python.
pub const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Element type of a stored tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// `F64`
    F64,
    /// `F32`
    F32,
    /// `F16`
    F16,
    /// `BF16`, read as float32 (exact widening).
    BF16,
    /// `I64`
    I64,
    /// `I32`
    I32,
    /// `I16`
    I16,
    /// `I8`
    I8,
    /// `U64`
    U64,
    /// `U32`
    U32,
    /// `U16`
    U16,
    /// `U8`
    U8,
    /// `BOOL`
    Bool,
    /// `F8_E4M3`, read as its raw `uint8` codes.
    F8E4M3,
    /// `F8_E8M0`, read as its raw `uint8` codes.
    F8E8M0,
}

impl Dtype {
    /// The dtype of a header tag, or `None` for a tag that the reader does not support.
    pub fn from_tag(tag: &str) -> Option<Dtype> {
        Some(match tag {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U64" => Dtype::U64,
            "U32" => Dtype::U32,
            "U16" => Dtype::U16,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E8M0" => Dtype::F8E8M0,
            _ => return None,
        })
    }

    /// The header tag.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U64 => "U64",
            Dtype::U32 => "U32",
            Dtype::U16 => "U16",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E8M0 => "F8_E8M0",
        }
    }

    /// Bytes per element in the file.
    pub fn itemsize(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool | Dtype::F8E4M3 | Dtype::F8E8M0 => 1,
        }
    }

    /// The NumPy dtype name of the array that the Python reader returns.
    pub fn numpy_name(self) -> &'static str {
        match self {
            Dtype::F64 => "float64",
            Dtype::F32 | Dtype::BF16 => "float32",
            Dtype::F16 => "float16",
            Dtype::I64 => "int64",
            Dtype::I32 => "int32",
            Dtype::I16 => "int16",
            Dtype::I8 => "int8",
            Dtype::U64 => "uint64",
            Dtype::U32 => "uint32",
            Dtype::U16 => "uint16",
            Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E8M0 => "uint8",
            Dtype::Bool => "bool",
        }
    }

    /// True when the Python reader returns a floating array (`np.issubdtype(dtype, np.floating)`).
    pub fn is_floating(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16)
    }
}

/// One tensor read from a file.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// The stored type.
    pub dtype: Dtype,
    /// The shape; empty for a scalar.
    pub shape: Vec<usize>,
    /// Little-endian bytes as stored.
    pub raw: Vec<u8>,
}

/// The value of one half-precision float.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 if frac == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// The float32 value of one bfloat16.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

impl Tensor {
    /// A tensor from float32 values (used by tests and callers that build arrays).
    pub fn from_f32(shape: Vec<usize>, values: &[f32]) -> Tensor {
        Tensor {
            dtype: Dtype::F32,
            shape,
            raw: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The NumPy dtype name of the Python array.
    pub fn numpy_dtype(&self) -> &'static str {
        self.dtype.numpy_name()
    }

    fn chunks(&self) -> std::slice::ChunksExact<'_, u8> {
        self.raw.chunks_exact(self.dtype.itemsize())
    }

    /// Every element as float64, as `astype(np.float64)` gives it.
    pub fn to_f64(&self) -> Vec<f64> {
        let c = self.chunks();
        match self.dtype {
            Dtype::F64 => c
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F32 => c
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            Dtype::F16 => c
                .map(|b| f16_to_f64(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            Dtype::BF16 => c
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
                .collect(),
            Dtype::I64 => c
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            Dtype::I32 => c
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            Dtype::I16 => c.map(|b| i16::from_le_bytes([b[0], b[1]]) as f64).collect(),
            Dtype::I8 => c.map(|b| b[0] as i8 as f64).collect(),
            Dtype::U64 => c
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            Dtype::U32 => c
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            Dtype::U16 => c.map(|b| u16::from_le_bytes([b[0], b[1]]) as f64).collect(),
            Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E8M0 => c.map(|b| b[0] as f64).collect(),
            Dtype::Bool => c.map(|b| if b[0] != 0 { 1.0 } else { 0.0 }).collect(),
        }
    }

    /// Every element of an integer or bool tensor, exactly. `None` for a float tensor.
    pub fn to_i128(&self) -> Option<Vec<i128>> {
        let c = self.chunks();
        Some(match self.dtype {
            Dtype::I64 => c
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i128)
                .collect(),
            Dtype::I32 => c
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i128)
                .collect(),
            Dtype::I16 => c
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i128)
                .collect(),
            Dtype::I8 => c.map(|b| b[0] as i8 as i128).collect(),
            Dtype::U64 => c
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as i128)
                .collect(),
            Dtype::U32 => c
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as i128)
                .collect(),
            Dtype::U16 => c
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as i128)
                .collect(),
            Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E8M0 => c.map(|b| b[0] as i128).collect(),
            Dtype::Bool => c.map(|b| (b[0] != 0) as i128).collect(),
            Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16 => return None,
        })
    }

    /// The float32 bit patterns of a tensor that Python reads as float32 (`F32` and `BF16`).
    pub fn f32_bits(&self) -> Option<Vec<u32>> {
        let c = self.chunks();
        match self.dtype {
            Dtype::F32 => Some(
                c.map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
            Dtype::BF16 => Some(
                c.map(|b| (u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The bytes of the NumPy array that Python returns (`a.tobytes()`): BF16 widened to float32, bool as 0/1.
    pub fn numpy_bytes(&self) -> Vec<u8> {
        match self.dtype {
            Dtype::BF16 => self
                .f32_bits()
                .unwrap()
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect(),
            _ => self.raw.clone(),
        }
    }

    /// Element `i` as Python's `.item()` gives it: an int, a bool or a float.
    pub fn item(&self, i: usize) -> Value {
        if self.dtype == Dtype::Bool {
            return Value::Bool(self.raw[i] != 0);
        }
        match self.to_i128() {
            Some(v) => {
                let x = v[i];
                if let Ok(s) = i64::try_from(x) {
                    s.into()
                } else {
                    (x as u64).into()
                }
            }
            None => crate::envelope::num(self.to_f64()[i]),
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// A lazy reader. Each tensor is read from disk only when requested.
#[derive(Clone, Debug)]
pub struct SafetensorsReader {
    /// The file.
    pub path: PathBuf,
    shown: String,
    base: u64,
    entries: BTreeMap<String, Entry>,
    /// `__metadata__`, or an empty object.
    pub metadata: Value,
}

fn invalid(msg: String) -> CumlxError {
    CumlxError::new("E_INPUT_INVALID", msg)
}

fn as_count(v: &Value) -> Option<i128> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from)),
        _ => None,
    }
}

impl SafetensorsReader {
    /// Open a file and check its header.
    pub fn open(path: &Path) -> CumlxResult<Self> {
        let shown = py_path(&path.to_string_lossy());
        if !path.is_file() {
            return Err(CumlxError::new(
                "E_INPUT_NOT_FOUND",
                format!("safetensors file not found: {shown}"),
            ));
        }
        let io = |e: std::io::Error| invalid(format!("{shown} could not be read: {e}"));
        let mut f = std::fs::File::open(path).map_err(io)?;
        let size = f.metadata().map_err(io)?.len();
        let mut head = [0u8; 8];
        let mut got = 0;
        while got < 8 {
            let n = f.read(&mut head[got..]).map_err(io)?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got != 8 {
            return Err(invalid(format!("{shown} is too short to be safetensors")));
        }
        let n = u64::from_le_bytes(head);
        if n > MAX_HEADER_BYTES || 8 + n > size {
            return Err(invalid(format!("{shown} has an invalid header length")));
        }
        let mut raw = vec![0u8; n as usize];
        f.read_exact(&mut raw).map_err(io)?;
        let header: Value = serde_json::from_slice(&raw)
            .map_err(|e| invalid(format!("{shown} header is not JSON: {e}")))?;
        let Value::Object(mut header) = header else {
            return Err(invalid(format!("{shown} header is not an object")));
        };
        let metadata = match header.shift_remove("__metadata__") {
            Some(v) if is_truthy(&v) => v,
            _ => Value::Object(Map::new()),
        };
        let base = 8 + n;
        let buffer_size = (size - base) as i128;
        let mut entries = BTreeMap::new();
        let mut spans = Vec::new();
        for (name, info) in &header {
            let malformed = || invalid(format!("{shown}: malformed entry {}", repr_str(name)));
            let tag = info.get("dtype").ok_or_else(malformed)?;
            let shape: Vec<i128> = info
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(malformed)?
                .iter()
                .map(as_count)
                .collect::<Option<_>>()
                .ok_or_else(malformed)?;
            let offsets = info
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(malformed)?;
            if offsets.len() != 2 {
                return Err(malformed());
            }
            let (begin, end) = match (as_count(&offsets[0]), as_count(&offsets[1])) {
                (Some(b), Some(e)) => (b, e),
                _ => return Err(malformed()),
            };
            let dtype = tag.as_str().and_then(Dtype::from_tag);
            let Some(dtype) = dtype.filter(|_| shape.iter().all(|s| *s >= 0)) else {
                return Err(invalid(format!(
                    "{shown}: unsupported dtype/shape for {}",
                    repr_str(name)
                )));
            };
            let expected = shape
                .iter()
                .try_fold(1i128, |acc, s| acc.checked_mul(*s))
                .and_then(|n| n.checked_mul(dtype.itemsize() as i128));
            if !(0 <= begin && begin <= end && end <= buffer_size) || Some(end - begin) != expected
            {
                return Err(invalid(format!(
                    "{shown}: offsets of {} do not match its shape",
                    repr_str(name)
                )));
            }
            let shape = shape.iter().map(|s| *s as usize).collect();
            entries.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    begin: begin as u64,
                    end: end as u64,
                },
            );
            spans.push((begin, end));
        }
        spans.sort();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(invalid(format!("{shown}: tensor byte ranges overlap")));
        }
        Ok(SafetensorsReader {
            path: path.to_path_buf(),
            shown,
            base,
            entries,
            metadata,
        })
    }

    /// Tensor names, sorted.
    pub fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// True when the file holds `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The stored type of `name`.
    pub fn dtype(&self, name: &str) -> Option<Dtype> {
        self.entries.get(name).map(|e| e.dtype)
    }

    /// The shape of `name`.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Read one tensor.
    pub fn get(&self, name: &str) -> CumlxResult<Tensor> {
        let Some(e) = self.entries.get(name) else {
            return Err(invalid(format!(
                "{}: tensor {} is missing",
                self.shown,
                repr_str(name)
            )));
        };
        let len = (e.end - e.begin) as usize;
        let mut raw = vec![0u8; len];
        if len > 0 {
            let truncated = || {
                invalid(format!(
                    "{}: truncated data for {}",
                    self.shown,
                    repr_str(name)
                ))
            };
            let mut f = std::fs::File::open(&self.path).map_err(|_| truncated())?;
            f.seek(SeekFrom::Start(self.base + e.begin))
                .map_err(|_| truncated())?;
            f.read_exact(&mut raw).map_err(|_| truncated())?;
        }
        Ok(Tensor {
            dtype: e.dtype,
            shape: e.shape.clone(),
            raw,
        })
    }
}

/// Python truthiness of a JSON value.
pub fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Write a safetensors file in the layout of the Python writer (sorted names, header padded to 8 bytes).
pub fn save_file(
    path: &Path,
    tensors: &BTreeMap<String, Tensor>,
    metadata: Option<&Map<String, Value>>,
) -> std::io::Result<()> {
    let mut header = Map::new();
    if let Some(m) = metadata {
        header.insert("__metadata__".into(), Value::Object(m.clone()));
    }
    let mut offset = 0usize;
    for (name, t) in tensors {
        let mut info = Map::new();
        info.insert("dtype".into(), t.dtype.tag().into());
        info.insert(
            "shape".into(),
            t.shape.iter().map(|s| Value::from(*s)).collect(),
        );
        info.insert(
            "data_offsets".into(),
            Value::Array(vec![offset.into(), (offset + t.raw.len()).into()]),
        );
        header.insert(name.clone(), Value::Object(info));
        offset += t.raw.len();
    }
    let mut raw = serde_json::to_vec(&Value::Object(header)).map_err(std::io::Error::other)?;
    while raw.len() % 8 != 0 {
        raw.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + raw.len() + offset);
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    out.extend_from_slice(&raw);
    for t in tensors.values() {
        out.extend_from_slice(&t.raw);
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_widen_exactly() {
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0xc000), -2.0);
        assert_eq!(f16_to_f64(0x0001), 2f64.powi(-24));
        assert_eq!(f16_to_f64(0x7bff), 65504.0);
        assert!(f16_to_f64(0x7c00).is_infinite());
        assert!(f16_to_f64(0x7e00).is_nan());
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn a_written_file_reads_back() {
        let dir = std::env::temp_dir().join(format!("cumlx-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "b".to_string(),
            Tensor::from_f32(vec![2, 2], &[1.0, -2.0, 3.5, 0.0]),
        );
        tensors.insert(
            "a".to_string(),
            Tensor {
                dtype: Dtype::I64,
                shape: vec![3],
                raw: [1i64, 5, 9].iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        );
        tensors.insert(
            "z".to_string(),
            Tensor {
                dtype: Dtype::F32,
                shape: vec![0, 4],
                raw: vec![],
            },
        );
        save_file(&path, &tensors, None).unwrap();
        let r = SafetensorsReader::open(&path).unwrap();
        assert_eq!(r.keys(), ["a", "b", "z"]);
        assert_eq!(r.get("b").unwrap().to_f64(), [1.0, -2.0, 3.5, 0.0]);
        assert_eq!(r.get("a").unwrap().to_i128().unwrap(), [1, 5, 9]);
        assert_eq!(r.get("z").unwrap().numel(), 0);
        assert_eq!(r.get("nope").unwrap_err().code, "E_INPUT_INVALID");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
