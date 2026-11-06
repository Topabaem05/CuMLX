//! The language-independent core of CUMLX, the verifier that an MLX port computes what the original
//! PyTorch or CUDA model computes.
//!
//! This crate reads golden traces (`cumlx.trace.v1`) and safetensors checkpoints, compares tensors with the
//! metrics and threshold profiles of the Python implementation, and writes the `cumlx.result.v1` envelope. It
//! needs neither torch nor MLX, so it cannot capture a golden or run a port: those stay in the Python `cumlx`.
//!
//! Every number and verdict here is meant to equal the Python one. `tests/test_rust_parity.py` in the
//! repository runs both implementations on the same inputs and fails on any difference larger than 1e-12
//! relative.

#![deny(missing_docs)]

pub mod envelope;
pub mod errors;
pub mod metrics;
pub mod pyfmt;
pub mod safetensors;
pub mod sha256;
pub mod trace;
pub mod trace_compare;
pub mod weights;

pub use envelope::Envelope;
pub use errors::{CumlxError, CumlxResult, ExitCode, ERROR_CODES};

/// The version of this crate, which follows the Python package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
