//! The result envelope `cumlx.result.v1`, as `src/cumlx/cli/output.py` writes it.
//!
//! * In JSON mode stdout carries exactly one JSON document and nothing else.
//! * `exit_code` in the document equals the process exit status.
//! * `status` is one of `ok | diverged | incomplete | error`.
//! * Long lists are capped by `--max-items`; `truncated` says when.

use crate::errors::{CumlxError, ExitCode};
use crate::pyfmt::{format_g, json_dumps, py_str, Dumps};
use serde_json::{Map, Value};

/// Schema name of the envelope.
pub const RESULT_SCHEMA: &str = "cumlx.result.v1";

/// The keys of the envelope, in order.
pub const ENVELOPE_KEYS: [&str; 14] = [
    "schema",
    "command",
    "ok",
    "status",
    "exit_code",
    "summary",
    "data",
    "artifacts",
    "warnings",
    "errors",
    "next_steps",
    "truncated",
    "cumlx_version",
    "elapsed_ms",
];

/// What one command returns before it is written.
#[derive(Clone, Debug)]
pub struct Envelope {
    /// The command name, such as `tracediff` or `kernel capture`.
    pub command: String,
    /// The process exit code.
    pub exit_code: ExitCode,
    /// One line for humans.
    pub summary: String,
    /// Command-specific data.
    pub data: Map<String, Value>,
    /// Files written: `{kind, path}`.
    pub artifacts: Vec<Value>,
    /// `{code, message, ...}`.
    pub warnings: Vec<Value>,
    /// `{code, message, hint?, details?}`.
    pub errors: Vec<Value>,
    /// `{command, why}`.
    pub next_steps: Vec<Value>,
    /// True when a list was cut by `--max-items`.
    pub truncated: bool,
}

impl Envelope {
    /// An empty successful result.
    pub fn new(command: impl Into<String>) -> Self {
        Envelope {
            command: command.into(),
            exit_code: ExitCode::Ok,
            summary: String::new(),
            data: Map::new(),
            artifacts: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            next_steps: Vec::new(),
            truncated: false,
        }
    }

    /// The result of a command that failed with `err`.
    pub fn from_error(command: impl Into<String>, err: &CumlxError) -> Self {
        let mut r = Envelope::new(command);
        r.exit_code = err.exit_code();
        r.summary = err.message.clone();
        r.errors.push(err.to_value());
        r
    }

    /// Add a warning.
    pub fn warn(&mut self, code: &str, message: &str) {
        let mut w = Map::new();
        w.insert("code".into(), code.into());
        w.insert("message".into(), message.into());
        self.warnings.push(Value::Object(w));
    }

    /// Suggest the next command.
    pub fn suggest(&mut self, command: &str, why: &str) {
        let mut s = Map::new();
        s.insert("command".into(), command.into());
        s.insert("why".into(), why.into());
        self.next_steps.push(Value::Object(s));
    }

    /// Name a written file.
    pub fn artifact(&mut self, kind: &str, path: &str) {
        let mut a = Map::new();
        a.insert("kind".into(), kind.into());
        a.insert("path".into(), path.into());
        self.artifacts.push(Value::Object(a));
    }

    /// `{items, total, omitted}`, capped at `max_items` (negative = all).
    pub fn cap(&mut self, items: Vec<Value>, max_items: i64) -> Value {
        let total = items.len();
        let mut out = Map::new();
        if max_items >= 0 && total as i64 > max_items {
            self.truncated = true;
            let keep = max_items as usize;
            out.insert(
                "items".into(),
                Value::Array(items.into_iter().take(keep).collect()),
            );
            out.insert("total".into(), total.into());
            out.insert("omitted".into(), (total - keep).into());
        } else {
            out.insert("items".into(), Value::Array(items));
            out.insert("total".into(), total.into());
            out.insert("omitted".into(), 0.into());
        }
        Value::Object(out)
    }

    /// The JSON document.
    pub fn to_value(&self, version: &str, elapsed_ms: u64) -> Value {
        let mut d = Map::new();
        d.insert("schema".into(), RESULT_SCHEMA.into());
        d.insert("command".into(), self.command.clone().into());
        d.insert("ok".into(), (self.exit_code == ExitCode::Ok).into());
        d.insert("status".into(), self.exit_code.status().into());
        d.insert("exit_code".into(), self.exit_code.code().into());
        d.insert("summary".into(), self.summary.clone().into());
        d.insert("data".into(), Value::Object(self.data.clone()));
        d.insert("artifacts".into(), Value::Array(self.artifacts.clone()));
        d.insert("warnings".into(), Value::Array(self.warnings.clone()));
        d.insert("errors".into(), Value::Array(self.errors.clone()));
        d.insert("next_steps".into(), Value::Array(self.next_steps.clone()));
        d.insert("truncated".into(), self.truncated.into());
        d.insert("cumlx_version".into(), version.into());
        d.insert("elapsed_ms".into(), elapsed_ms.into());
        Value::Object(d)
    }
}

/// A float for JSON: non-finite values become `null`, as `_sanitize` does in Python.
pub fn num(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// An optional float for JSON.
pub fn opt_num(x: Option<f64>) -> Value {
    x.map(num).unwrap_or(Value::Null)
}

/// The document as `json.dumps(doc, indent=2, ensure_ascii=False)` writes it.
pub fn to_json(doc: &Value) -> String {
    json_dumps(
        doc,
        Dumps {
            indent: Some(2),
            ensure_ascii: false,
        },
    )
}

fn text_value(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_f64() => format_g(n.as_f64().unwrap_or(f64::NAN), 4),
        Value::Array(_) | Value::Object(_) => {
            let s = json_dumps(
                v,
                Dumps {
                    indent: None,
                    ensure_ascii: false,
                },
            );
            if s.chars().count() <= 160 {
                s
            } else {
                format!("{}...", s.chars().take(157).collect::<String>())
            }
        }
        other => py_str(other),
    }
}

fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or(&Value::Null)
}

/// The text rendering of `to_text` in Python, for a human at a terminal. `tool` is `cumlx` in Python.
pub fn to_text(doc: &Value, tool: &str) -> String {
    let mut lines = vec![format!(
        "[{}] {tool} {}: {}",
        py_str(field(doc, "status")).to_uppercase(),
        py_str(field(doc, "command")),
        py_str(field(doc, "summary"))
    )];
    if let Some(data) = doc.get("data").and_then(Value::as_object) {
        for (k, v) in data {
            lines.push(format!("  {k}: {}", text_value(v)));
        }
    }
    let list = |key: &str| {
        doc.get(key)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    for a in list("artifacts") {
        lines.push(format!(
            "  artifact ({}): {}",
            py_str(field(&a, "kind")),
            py_str(field(&a, "path"))
        ));
    }
    for w in list("warnings") {
        lines.push(format!(
            "  warning {}: {}",
            py_str(field(&w, "code")),
            py_str(field(&w, "message"))
        ));
    }
    for e in list("errors") {
        lines.push(format!(
            "  error {}: {}",
            py_str(field(&e, "code")),
            py_str(field(&e, "message"))
        ));
        if let Some(h) = e
            .get("hint")
            .filter(|h| !h.is_null() && h.as_str() != Some(""))
        {
            lines.push(format!("    hint: {}", py_str(h)));
        }
    }
    for s in list("next_steps") {
        lines.push(format!(
            "  next: {}    # {}",
            py_str(field(&s, "command")),
            py_str(field(&s, "why"))
        ));
    }
    if doc
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        lines.push("  (lists truncated by --max-items; see artifacts for full data)".into());
    }
    lines.push(format!("  exit_code={}", py_str(field(doc, "exit_code"))));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cap_marks_truncation() {
        let mut r = Envelope::new("x");
        let v = r.cap(vec![json!(1), json!(2), json!(3)], 2);
        assert_eq!(v, json!({"items": [1, 2], "total": 3, "omitted": 1}));
        assert!(r.truncated);
        let mut r = Envelope::new("x");
        assert_eq!(
            r.cap(vec![json!(1)], -1),
            json!({"items": [1], "total": 1, "omitted": 0})
        );
        assert!(!r.truncated);
    }

    #[test]
    fn non_finite_floats_become_null() {
        assert_eq!(num(f64::INFINITY), Value::Null);
        assert_eq!(num(0.5), json!(0.5));
    }

    #[test]
    fn the_envelope_has_every_key_in_order() {
        let mut r = Envelope::new("tracediff");
        r.exit_code = ExitCode::Incomplete;
        let doc = r.to_value("0.5.0", 3);
        let keys: Vec<&str> = doc
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ENVELOPE_KEYS);
        assert_eq!(doc["status"], "incomplete");
        assert_eq!(doc["ok"], false);
        assert!(to_text(&doc, "cumlx").starts_with("[INCOMPLETE] cumlx tracediff: "));
    }
}
