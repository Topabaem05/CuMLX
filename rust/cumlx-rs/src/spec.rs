//! The command table. `describe` and the argument parser both read it, so the description cannot drift from
//! the parser, as in the Python CLI.

use serde_json::Value;

/// How an argument takes values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A switch without a value.
    Flag,
    /// One value.
    Value,
}

/// The type an argument value converts to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// Text.
    Str,
    /// `int(...)`.
    Int,
    /// `float(...)`.
    Float,
    /// A flag.
    Bool,
}

impl Ty {
    pub fn name(self) -> &'static str {
        match self {
            Ty::Str => "str",
            Ty::Int => "int",
            Ty::Float => "float",
            Ty::Bool => "bool",
        }
    }
}

/// One argument.
#[derive(Clone, Debug)]
pub struct Arg {
    /// Destination name, as argparse's `dest`.
    pub name: &'static str,
    /// Option strings; empty for a positional.
    pub flags: &'static [&'static str],
    pub kind: Kind,
    pub ty: Ty,
    pub default: Value,
    pub choices: Option<Vec<String>>,
    pub help: &'static str,
}

impl Arg {
    pub fn positional(&self) -> bool {
        self.flags.is_empty()
    }
}

/// One command of the Python CLI, implemented here or not.
#[derive(Clone, Debug)]
pub struct Command {
    pub name: &'static str,
    pub summary: &'static str,
    pub requires: &'static [&'static str],
    pub stage: &'static str,
    pub examples: &'static [&'static str],
    /// `None` when only the Python cumlx runs it.
    pub args: Option<Vec<Arg>>,
    /// What it needs that this binary does not have, for a command that it does not run.
    pub needs: &'static str,
    /// The install command of the Python cumlx for this command.
    pub install: &'static str,
}

fn env_default(var: &str, fallback: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| fallback.to_string())
}

fn common() -> Vec<Arg> {
    let max_items: Value = match env_default("CUMLX_MAX_ITEMS", "20").trim().parse::<i64>() {
        Ok(n) => n.into(),
        Err(_) => Value::from(env_default("CUMLX_MAX_ITEMS", "20")),
    };
    vec![
        Arg {
            name: "format",
            flags: &["--format"],
            kind: Kind::Value,
            ty: Ty::Str,
            default: env_default("CUMLX_FORMAT", "auto").into(),
            choices: Some(vec!["auto".into(), "json".into(), "text".into()]),
            help: "json = one JSON document on stdout; auto = json when stdout is not a TTY (env CUMLX_FORMAT).",
        },
        Arg {
            name: "max_items",
            flags: &["--max-items"],
            kind: Kind::Value,
            ty: Ty::Int,
            default: max_items,
            choices: None,
            help: "cap list lengths in stdout (-1 = unlimited; env CUMLX_MAX_ITEMS). Default 20.",
        },
        Arg {
            name: "verbose",
            flags: &["-v", "--verbose"],
            kind: Kind::Flag,
            ty: Ty::Bool,
            default: false.into(),
            choices: None,
            help: "progress logs on stderr",
        },
    ]
}

fn positional(name: &'static str, help: &'static str) -> Arg {
    Arg {
        name,
        flags: &[],
        kind: Kind::Value,
        ty: Ty::Str,
        default: Value::Null,
        choices: None,
        help,
    }
}

fn option(
    name: &'static str,
    flags: &'static [&'static str],
    ty: Ty,
    default: Value,
    help: &'static str,
) -> Arg {
    let kind = if ty == Ty::Bool {
        Kind::Flag
    } else {
        Kind::Value
    };
    Arg {
        name,
        flags,
        kind,
        ty,
        default,
        choices: None,
        help,
    }
}

fn with(mut args: Vec<Arg>, more: Vec<Arg>) -> Option<Vec<Arg>> {
    args.extend(more);
    Some(args)
}

/// Every command name, sorted, as `--command` accepts them.
pub fn names() -> Vec<String> {
    let mut n: Vec<String> = commands_without_describe_choices()
        .iter()
        .map(|c| c.name.to_string())
        .collect();
    n.sort();
    n
}

/// The full table.
pub fn commands() -> Vec<Command> {
    let mut all = commands_without_describe_choices();
    let choices = names();
    if let Some(args) = all
        .iter_mut()
        .find(|c| c.name == "describe")
        .and_then(|c| c.args.as_mut())
    {
        if let Some(a) = args.iter_mut().find(|a| a.name == "target_command") {
            a.choices = Some(choices);
        }
    }
    all
}

fn python_only(
    name: &'static str,
    summary: &'static str,
    requires: &'static [&'static str],
    stage: &'static str,
    examples: &'static [&'static str],
    needs: &'static str,
    install: &'static str,
) -> Command {
    Command {
        name,
        summary,
        requires,
        stage,
        examples,
        args: None,
        needs,
        install,
    }
}

const BASE: &str = "python -m pip install cumlx-lab";
const CAPTURE: &str = "python -m pip install 'cumlx-lab[capture]'";
const DIFF: &str = "python -m pip install 'cumlx-lab[diff]'";
const BOTH: &str = "python -m pip install 'cumlx-lab[capture,diff]'";

fn commands_without_describe_choices() -> Vec<Command> {
    vec![
        Command {
            name: "describe",
            summary: "Print the machine-readable specification of cumlx-rs: commands, arguments, exit and error codes, and which commands only the Python cumlx runs.",
            requires: &[],
            stage: "stable",
            examples: &["cumlx-rs describe", "cumlx-rs describe --command tracediff"],
            args: with(common(), vec![Arg {
                name: "target_command",
                flags: &["--command"],
                kind: Kind::Value,
                ty: Ty::Str,
                default: Value::Null,
                choices: None,
                help: "describe only this command",
            }]),
            needs: "",
            install: "",
        },
        python_only("doctor", "Report installed dependencies and which commands are ready; with --require or --stack, fail when not ready.", &[], "stable",
            &["cumlx doctor", "cumlx doctor --probe --require capture,diff", "cumlx doctor --stack"],
            "a Python environment to inspect", BASE),
        python_only("fixture", "Create a tiny random HF model (safetensors) for offline end-to-end tests.", &["capture"], "beta",
            &["cumlx fixture tiny-llama -o /tmp/tiny", "cumlx fixture tiny-moe -o /tmp/tiny-moe"],
            "torch and transformers", CAPTURE),
        python_only("capture", "Run the original PyTorch model and record per-module inputs/outputs as a golden trace.", &["capture"], "beta",
            &["cumlx capture /tmp/tiny --token-ids 1,5,9,23 -o /tmp/golden",
              "cumlx capture Qwen/Qwen2.5-0.5B --prompt 'Hello' -o golden/",
              "cumlx capture Qwen/Qwen2.5-0.5B --prompt 'Hello' --weights-from-mlx ./qwen-4bit -o golden-q/"],
            "torch and transformers", CAPTURE),
        python_only("diff", "Run an MLX port on the trace inputs, compare every module and locate the first divergence.", &["diff"], "beta",
            &["cumlx diff /tmp/golden /tmp/tiny -o /tmp/report", "cumlx diff golden/ ./qwen-mlx --device cpu", "cumlx diff golden/ ./qwen-mlx --tf32 on"],
            "MLX and mlx-lm, to run the port", DIFF),
        python_only("conform", "Check FP4 (NVFP4/MXFP4) decode conformance of a target against the CUMLX bit-exact codec.", &["diff"], "beta",
            &["cumlx conform --target mlx", "cumlx conform --target mlx --formats mxfp4",
              "cumlx conform --checkpoint nvidia/model-NVFP4 --against someone/model-MLX-NVFP4"],
            "MLX, and the FP4 codec, which is not ported", DIFF),
        Command {
            name: "weights",
            summary: "Compare the weights of an unquantized port, read from its model*.safetensors files, with the original safetensors without running a model (fast; names transposes and scale errors).",
            requires: &[],
            stage: "beta",
            examples: &["cumlx-rs weights ./original ./port-mlx", "cumlx-rs weights /tmp/tiny /tmp/tiny -o weights.json"],
            args: with(common(), vec![
                positional("original", "local directory or safetensors file of the ORIGINAL checkpoint (a Hugging Face id needs the Python cumlx)"),
                positional("port", "local port directory in mlx-lm format: config.json and model*.safetensors, unquantized"),
                option("output", &["-o", "--output"], Ty::Str, Value::Null, "write every parameter to this JSON file"),
                option("loader", &["--loader"], Ty::Str, "mlx_lm".into(), "only 'mlx_lm' (read the files that mlx-lm loads); a Python loader needs the Python cumlx"),
                option("mapping", &["--mapping"], Ty::Str, Value::Null, "name mapping rules; only the Python cumlx applies them"),
                option("revision", &["--revision"], Ty::Str, Value::Null, "HF revision; ignored for a local checkpoint, as in Python"),
                option("overwrite", &["--overwrite"], Ty::Bool, false.into(), "replace an existing --output file"),
            ]),
            needs: "",
            install: "",
        },
        python_only("scan", "Triage a CUDA repository or an HF model: read its source as text and rank the work to run it on Apple Silicon.", &[], "beta",
            &["cumlx scan ./some-cuda-repo", "cumlx scan Qwen/Qwen3-0.6B"],
            "nothing that Rust lacks; it is not ported", BASE),
        python_only("recipe", "Name what to use on Apple Silicon for a CUDA dependency, and show whether each option was run or only read about.", &[], "beta",
            &["cumlx recipe", "cumlx recipe bitsandbytes", "cumlx recipe ./some-cuda-repo --verified-only", "cumlx recipe scan.json"],
            "nothing that Rust lacks; it is not ported", BASE),
        python_only("run", "Run an unmodified PyTorch script that assumes CUDA on MPS, and record each redirected call. Running is not proof.", &["run"], "experimental",
            &["cumlx run train.py --target mps --report run-report.json", "cumlx run train.py -- --epochs 1 --lr 1e-3",
              "cumlx run --module my.pkg.train", "cumlx run train.py --substitute flash_attn=mps_flash_attn",
              "cumlx run train.py --trace-losses --seed 1 --report mps/seed1.json"],
            "torch, to run the script", CAPTURE),
        python_only("rundiff", "Compare the loss traces of `cumlx run` on CUDA and on the target, and give a proof grade.", &[], "experimental",
            &["cumlx rundiff golden-cuda/ runs-mps/", "cumlx rundiff cuda/seed0.json mps/seed0.json"],
            "nothing that Rust lacks; it is not ported", BASE),
        python_only("port", "Convert a model with mlx-lm and prove the result: convert, compare weights, capture a golden, diff.", &["capture", "diff"], "beta",
            &["cumlx port Qwen/Qwen2.5-0.5B -o port/", "cumlx port Qwen/Qwen2.5-0.5B --q-bits 4 -o port-4bit/ --prompt Hello"],
            "mlx-lm to convert, torch to capture and MLX to diff", BOTH),
        Command {
            name: "tracediff",
            summary: "Compare two golden traces of one model, such as a CUDA capture and a CPU capture.",
            requires: &[],
            stage: "beta",
            examples: &["cumlx-rs tracediff golden-cuda/ golden-cpu/"],
            args: with(common(), vec![
                positional("a", "first trace directory"),
                positional("b", "second trace directory"),
                option("rel_l2", &["--rel-l2"], Ty::Float, 1e-4.into(), "report outputs that differ by more than this"),
            ]),
            needs: "",
            install: "",
        },
        python_only("report", "Summarize an existing diff report.json (re-read with a different --max-items).", &[], "beta",
            &["cumlx report /tmp/report/report.json --max-items 50"],
            "nothing that Rust lacks; it is not ported", BASE),
        python_only("selftest", "Prove the verifier works here: clean port must pass and injected faults must be located.", &[], "beta",
            &["cumlx selftest", "cumlx selftest --core-only"],
            "torch and MLX for its end-to-end checks; `cargo test` and tests/test_rust_parity.py prove this binary", BOTH),
        python_only("kernel capture", "Compile the original CUDA kernels with nvcc, run them on a real NVIDIA GPU and record an op-level golden. Executes the code.", &["nvcc"], "experimental",
            &["cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --require-idle-gpu",
              "cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --dry-run"],
            "nvcc and an NVIDIA GPU", BASE),
        python_only("kernel verify", "Run a Metal kernel on the inputs of an op-level golden and compare the outputs. An unknown or unmapped kernel is never a pass.", &["metal"], "experimental",
            &["cumlx kernel verify foo.metal kgolden/", "cumlx kernel verify kernels/ kgolden/ -o report/",
              "cumlx kernel verify translated.metal kgolden/ --kernel foo --engine torch"],
            "MLX or torch to run a Metal kernel", "python -m pip install 'cumlx-lab[metal]'"),
        python_only("kernel port", "Prepare a .cu file for translation (headers, templates, host code, PTX built-ins) and call a translator. A feature with no Metal equivalent is refused.", &[], "experimental",
            &["cumlx kernel port foo.cu -o foo.metal", "cumlx kernel port foo.cu --preprocess-only -o foo.pre.cu",
              "cumlx kernel port foo.cu -o foo.metal --instantiate \"foo<float>\""],
            "nothing that Rust lacks; it is not ported", BASE),
    ]
}
