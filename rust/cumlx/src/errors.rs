//! Stable error codes and exit codes, the same table as `ERROR_CODES` in `src/cumlx/errors.py`.
//!
//! `tests/test_rust_parity.py` fails when the two tables differ.

use serde_json::{Map, Value};
use std::fmt;

/// Process exit status. The JSON document carries the same number in `exit_code`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExitCode {
    /// Command succeeded; for verification: every compared item passed.
    Ok = 0,
    /// Verification ran and at least one item exceeded its threshold.
    Diverged = 1,
    /// Invalid arguments or an unsafe request (for example, output exists).
    Usage = 2,
    /// A dependency or backend is missing.
    Environment = 3,
    /// An input path, model or trace is missing, corrupt or has the wrong schema.
    Input = 4,
    /// Ran without divergence, but some items were not verified. Not a pass.
    Incomplete = 5,
    /// Unexpected internal error: a CUMLX bug.
    Internal = 10,
}

impl ExitCode {
    /// Every exit code, in the order of the Python enum.
    pub const ALL: [ExitCode; 7] = [
        ExitCode::Ok,
        ExitCode::Diverged,
        ExitCode::Usage,
        ExitCode::Environment,
        ExitCode::Input,
        ExitCode::Incomplete,
        ExitCode::Internal,
    ];

    /// The number given to the operating system.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The enum member name in Python (`ExitCode.OK.name`).
    pub fn name(self) -> &'static str {
        match self {
            ExitCode::Ok => "OK",
            ExitCode::Diverged => "DIVERGED",
            ExitCode::Usage => "USAGE",
            ExitCode::Environment => "ENVIRONMENT",
            ExitCode::Input => "INPUT",
            ExitCode::Incomplete => "INCOMPLETE",
            ExitCode::Internal => "INTERNAL",
        }
    }

    /// `EXIT_CODE_DOCS` in Python.
    pub fn meaning(self) -> &'static str {
        match self {
            ExitCode::Ok => "Success. For diff/conform: every compared item passed.",
            ExitCode::Diverged => {
                "Verification ran and found at least one divergence. Read data.first_divergence."
            }
            ExitCode::Usage => {
                "Invalid arguments or unsafe request. Fix the command; see errors[].hint."
            }
            ExitCode::Environment => {
                "A dependency or backend is missing. Run the install command in errors[].hint."
            }
            ExitCode::Input => {
                "An input path, model or trace is missing, corrupt or has the wrong schema."
            }
            ExitCode::Incomplete => {
                "No divergence, but some items were not verified. This is NOT a pass."
            }
            ExitCode::Internal => "Unexpected internal error. Report it with the full JSON output.",
        }
    }

    /// `status` in the result envelope.
    pub fn status(self) -> &'static str {
        match self {
            ExitCode::Ok => "ok",
            ExitCode::Diverged => "diverged",
            ExitCode::Incomplete => "incomplete",
            _ => "error",
        }
    }

    /// The exit code of a verdict string (`ok`, `diverged`, `incomplete`).
    pub fn from_verdict(verdict: &str) -> ExitCode {
        match verdict {
            "ok" => ExitCode::Ok,
            "diverged" => ExitCode::Diverged,
            "incomplete" => ExitCode::Incomplete,
            _ => ExitCode::Internal,
        }
    }
}

/// `ERROR_CODES`: code, exit code and a short description. The order is the Python order.
pub const ERROR_CODES: &[(&str, ExitCode, &str)] = &[
    ("E_USAGE", ExitCode::Usage, "Invalid command-line arguments."),
    ("E_OUTPUT_EXISTS", ExitCode::Usage, "Output path already exists; pass --overwrite to replace it."),
    ("E_DEP_MISSING", ExitCode::Environment, "An optional Python dependency is not installed."),
    ("E_BACKEND_UNAVAILABLE", ExitCode::Environment, "The requested compute backend is not available."),
    ("E_PYTHON_ONLY", ExitCode::Environment, "Only the Python cumlx runs this command or option; cumlx-rs (Rust) does not. errors[].hint gives the Python command."),
    ("E_INPUT_NOT_FOUND", ExitCode::Input, "An input path does not exist."),
    ("E_INPUT_INVALID", ExitCode::Input, "An input file is corrupt or does not match its schema."),
    ("E_MODEL_LOAD", ExitCode::Input, "The model could not be loaded."),
    ("E_MODEL_RUN", ExitCode::Input, "The model failed while running on the recorded inputs."),
    ("E_RUN_SCRIPT_FAILED", ExitCode::Input, "The script given to `cumlx run` failed or ran past its timeout."),
    ("E_RUN_UNSUPPORTED", ExitCode::Environment, "The script called a CUDA API with no equivalent on the target."),
    ("E_RUN_MPS_OP_MISSING", ExitCode::Environment, "The script used an operation that MPS does not implement."),
    ("E_RUN_MPS_DTYPE", ExitCode::Environment, "The script used a data type that MPS does not have."),
    ("E_RUN_NOT_COMPARABLE", ExitCode::Input, "These run reports are not runs of one script with one set of arguments, or one of them failed."),
    ("E_RUN_MISSING_PACKAGE", ExitCode::Environment, "The script imports a package that is not installed."),
    ("E_KERNEL_NO_CUDA", ExitCode::Environment, "`cumlx kernel capture` needs an NVIDIA GPU, nvcc and nvidia-smi."),
    ("E_KERNEL_GPU_BUSY", ExitCode::Environment, "The GPU has other compute processes, and --require-idle-gpu was given."),
    ("E_KERNEL_COMPILE", ExitCode::Input, "The compiler rejected a kernel or its generated program."),
    ("E_KERNEL_TRANSLATOR_MISSING", ExitCode::Environment, "The CUDA-to-Metal translator is not installed or cannot run. Nothing was translated."),
    ("E_KERNEL_TRANSLATE_FAILED", ExitCode::Input, "The translator failed on this kernel or returned an unusable result."),
    ("E_KERNEL_UNPORTABLE", ExitCode::Input, "The kernel uses a feature with no Metal equivalent. errors[].details.refused names each one and its reason."),
    ("E_SIM_UNSUPPORTED", ExitCode::Input, "This simulation engine cannot load this kind of asset."),
    ("E_RENDER_USD_UNSUPPORTED", ExitCode::Input, "A USD scene holds something that cumlx.render does not convert. Nothing is rendered."),
    ("E_SIZE_LIMIT", ExitCode::Usage, "The operation would exceed a configured size limit."),
    ("E_INTERNAL", ExitCode::Internal, "Unexpected internal error."),
];

/// The exit code of a known error code.
pub fn exit_code_of(code: &str) -> Option<ExitCode> {
    ERROR_CODES
        .iter()
        .find(|(c, _, _)| *c == code)
        .map(|(_, e, _)| *e)
}

/// An actionable error with a stable code. Never used for verification verdicts.
#[derive(Clone, Debug, PartialEq)]
pub struct CumlxError {
    /// A key of [`ERROR_CODES`].
    pub code: &'static str,
    /// One sentence.
    pub message: String,
    /// The exact next action, when there is one.
    pub hint: Option<String>,
    /// Machine-readable context, boxed to keep `Result<_, CumlxError>` small.
    pub details: Box<Map<String, Value>>,
}

impl CumlxError {
    /// A new error. Panics on a code that is not in [`ERROR_CODES`], as the Python constructor raises.
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        assert!(exit_code_of(code).is_some(), "unknown error code {code:?}");
        CumlxError {
            code,
            message: message.into(),
            hint: None,
            details: Box::default(),
        }
    }

    /// The same error with a hint.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// The same error with one more detail.
    pub fn detail(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.details.insert(key.to_string(), value.into());
        self
    }

    /// The process exit code for this error.
    pub fn exit_code(&self) -> ExitCode {
        exit_code_of(self.code).expect("checked in new()")
    }

    /// The JSON object in `errors[]`.
    pub fn to_value(&self) -> Value {
        let mut out = Map::new();
        out.insert("code".into(), self.code.into());
        out.insert("message".into(), self.message.clone().into());
        if let Some(h) = self.hint.as_ref().filter(|h| !h.is_empty()) {
            out.insert("hint".into(), h.clone().into());
        }
        if !self.details.is_empty() {
            out.insert("details".into(), Value::Object((*self.details).clone()));
        }
        Value::Object(out)
    }
}

impl fmt::Display for CumlxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CumlxError {}

/// Result type of this crate.
pub type CumlxResult<T> = std::result::Result<T, CumlxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_is_unique_and_resolves() {
        let mut seen = std::collections::HashSet::new();
        for (code, exit, _) in ERROR_CODES {
            assert!(seen.insert(*code), "{code} twice");
            assert_eq!(exit_code_of(code), Some(*exit));
        }
        assert_eq!(
            CumlxError::new("E_PYTHON_ONLY", "x").exit_code(),
            ExitCode::Environment
        );
    }

    #[test]
    #[should_panic(expected = "unknown error code")]
    fn an_unknown_code_is_a_bug() {
        let _ = CumlxError::new("E_NOPE", "x");
    }
}
