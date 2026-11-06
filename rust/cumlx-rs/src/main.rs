//! `cumlx-rs`: the Rust command line of the CUMLX core.
//!
//! The contract is the Python `cumlx` contract: with `--format json` (the default when stdout is not a TTY),
//! stdout is exactly one `cumlx.result.v1` document, logs go to stderr, and the exit codes and error codes are
//! the same. `tracediff`, `weights` and `describe` run here. Every other command of the Python CLI needs torch,
//! MLX, nvcc or code that is not ported, so it ends with `E_PYTHON_ONLY` (exit 3) and a hint that names the
//! Python command. It never runs a part of such a command.

mod args;
mod spec;

use cumlx::envelope::{num, to_json, to_text, Envelope, ENVELOPE_KEYS};
use cumlx::errors::{CumlxError, ExitCode, ERROR_CODES};
use cumlx::pyfmt::{json_dumps, shlex_quote, Dumps};
use cumlx::trace_compare::{compare_traces, summary_line};
use cumlx::weights::{python_only, run_weights};
use serde_json::{Map, Value};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

const TOOL: &str = "cumlx-rs";

fn detect_format(argv: &[String]) -> &'static str {
    let mut fmt = std::env::var("CUMLX_FORMAT").unwrap_or_else(|_| "auto".into());
    for (i, a) in argv.iter().enumerate() {
        if a == "--format" && i + 1 < argv.len() {
            fmt = argv[i + 1].clone();
        } else if let Some(v) = a.strip_prefix("--format=") {
            fmt = v.to_string();
        }
    }
    match fmt.as_str() {
        "json" => "json",
        "text" => "text",
        _ => {
            if std::io::stdout().is_terminal() {
                "text"
            } else {
                "json"
            }
        }
    }
}

/// The command that an argument list names: `kernel capture`, `tracediff`, ... or the tool itself.
fn command_name(argv: &[String]) -> String {
    let names = spec::names();
    for (i, a) in argv.iter().enumerate() {
        if a == "kernel" {
            let full = format!(
                "kernel {}",
                argv.get(i + 1).map(String::as_str).unwrap_or("")
            );
            if names.contains(&full) {
                return full;
            }
        }
        if names.contains(a) {
            return a.clone();
        }
    }
    TOOL.into()
}

fn usage(name: &str, message: String) -> Envelope {
    let mut r = Envelope::new(name);
    r.exit_code = ExitCode::Usage;
    r.summary = format!("invalid arguments: {message}");
    let hint = if spec::names().iter().any(|n| n == name) {
        format!("run `{TOOL} describe --command {name}`")
    } else {
        format!("run `{TOOL} describe`")
    };
    r.errors
        .push(CumlxError::new("E_USAGE", message).hint(hint).to_value());
    r
}

fn python_command(argv: &[String]) -> String {
    let mut words = vec!["cumlx".to_string()];
    words.extend(argv.iter().map(|a| shlex_quote(a)));
    words.join(" ")
}

fn log(args: &Map<String, Value>, message: &str) {
    if args
        .get("verbose")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        eprintln!("[{TOOL}] {message}");
    }
}

fn max_items(args: &Map<String, Value>) -> i64 {
    args["max_items"].as_i64().unwrap_or(20)
}

fn describe(args: &Map<String, Value>) -> Result<Envelope, CumlxError> {
    let all = spec::commands();
    let mut commands = Map::new();
    let mut python_only_names = Vec::new();
    for c in &all {
        let mut m = Map::new();
        m.insert("summary".into(), c.summary.into());
        m.insert(
            "requires".into(),
            c.requires.iter().map(|r| Value::from(*r)).collect(),
        );
        m.insert("stage".into(), c.stage.into());
        m.insert(
            "examples".into(),
            c.examples.iter().map(|r| Value::from(*r)).collect(),
        );
        m.insert("implemented".into(), c.args.is_some().into());
        match &c.args {
            Some(spec_args) => {
                let arguments: Vec<Value> = spec_args
                    .iter()
                    .map(|a| {
                        let mut d = Map::new();
                        d.insert("name".into(), a.name.into());
                        d.insert(
                            "flags".into(),
                            a.flags.iter().map(|f| Value::from(*f)).collect(),
                        );
                        d.insert("positional".into(), a.positional().into());
                        d.insert(
                            "kind".into(),
                            if a.kind == spec::Kind::Flag {
                                "flag"
                            } else {
                                "value"
                            }
                            .into(),
                        );
                        d.insert("type".into(), a.ty.name().into());
                        d.insert("required".into(), a.positional().into());
                        d.insert("default".into(), a.default.clone());
                        d.insert(
                            "choices".into(),
                            a.choices
                                .as_ref()
                                .map(|c| c.iter().map(|x| Value::from(x.as_str())).collect())
                                .unwrap_or(Value::Null),
                        );
                        d.insert("help".into(), a.help.into());
                        Value::Object(d)
                    })
                    .collect();
                m.insert("arguments".into(), Value::Array(arguments));
            }
            None => {
                python_only_names.push(Value::from(c.name));
                m.insert("needs".into(), c.needs.into());
                m.insert("python_command".into(), format!("cumlx {}", c.name).into());
                m.insert("install".into(), c.install.into());
            }
        }
        commands.insert(c.name.into(), Value::Object(m));
    }
    if let Some(target) = args.get("target_command").and_then(Value::as_str) {
        let only = commands.remove(target).expect("checked by choices");
        commands = Map::new();
        commands.insert(target.into(), only);
    }
    let mut r = Envelope::new("describe");
    r.summary = format!(
        "{} command(s); stdout schema cumlx.result.v1",
        commands.len()
    );
    let mut exit_codes = Map::new();
    for e in ExitCode::ALL {
        let mut d = Map::new();
        d.insert("name".into(), e.name().into());
        d.insert("meaning".into(), e.meaning().into());
        exit_codes.insert(e.code().to_string(), Value::Object(d));
    }
    let mut error_codes = Map::new();
    for (code, exit, meaning) in ERROR_CODES {
        let mut d = Map::new();
        d.insert("exit_code".into(), exit.code().into());
        d.insert("meaning".into(), (*meaning).into());
        error_codes.insert((*code).into(), Value::Object(d));
    }
    let mut env = Map::new();
    env.insert("CUMLX_FORMAT".into(), "auto|json|text".into());
    env.insert("CUMLX_MAX_ITEMS".into(), "integer, -1 = unlimited".into());
    let rules = [
        "Parse stdout as one JSON document; never scrape text output.",
        "Branch on exit_code: 0 ok, 1 diverged, 5 incomplete (not a pass), 2/3/4 fix input or environment.",
        "E_PYTHON_ONLY (exit 3) means this binary does not run the command or option: run errors[].hint, the same command with the Python cumlx.",
        "Never treat `incomplete` or an unverified parameter as success.",
        "Run `cumlx-rs weights` before `cumlx diff`: it takes seconds and names transposed or rescaled weights.",
        "`cumlx-rs weights` reads the port's model*.safetensors as stored and refuses a quantized port; it does not prove that mlx-lm loads the port.",
        "Use --max-items to bound output size.",
    ];
    let d = &mut r.data;
    d.insert("tool".into(), TOOL.into());
    d.insert("version".into(), cumlx::VERSION.into());
    d.insert("implementation".into(), "rust".into());
    d.insert(
        "purpose".into(),
        "The language-independent core of CUMLX in Rust: read and check golden traces, compare two traces, and compare \
         the weights of an unquantized port with the original. Capture, diff and kernels need the Python cumlx."
            .into(),
    );
    d.insert(
        "agent_rules".into(),
        rules.iter().map(|x| Value::from(*x)).collect(),
    );
    d.insert(
        "workflow".into(),
        Value::Array(vec![
            "cumlx-rs weights <original-dir> <mlx-port-dir>".into(),
            "cumlx capture <original> --prompt TEXT -o golden/   (Python)".into(),
            "cumlx diff golden/ <mlx-port> -o report/   (Python)".into(),
            "cumlx-rs tracediff golden-cuda/ golden-cpu/".into(),
        ]),
    );
    d.insert("commands".into(), Value::Object(commands));
    d.insert("python_only".into(), Value::Array(python_only_names));
    d.insert("exit_codes".into(), Value::Object(exit_codes));
    d.insert("error_codes".into(), Value::Object(error_codes));
    d.insert("env".into(), Value::Object(env));
    d.insert(
        "result_envelope".into(),
        ENVELOPE_KEYS.iter().map(|k| Value::from(*k)).collect(),
    );
    r.suggest(
        "cumlx doctor",
        "check which Python commands are ready in this environment",
    );
    Ok(r)
}

fn tracediff(args: &Map<String, Value>) -> Result<Envelope, CumlxError> {
    let rel_l2 = args::get_float(args, "rel_l2");
    if rel_l2.is_nan() || rel_l2 <= 0.0 {
        return Err(CumlxError::new("E_USAGE", "--rel-l2 must be positive"));
    }
    let (a, b) = (args["a"].as_str().unwrap(), args["b"].as_str().unwrap());
    log(args, &format!("comparing {a} with {b}"));
    let rep = compare_traces(a, b, rel_l2)?.report;
    let mut r = Envelope::new("tracediff");
    r.exit_code = ExitCode::from_verdict(rep["verdict"].as_str().unwrap());
    r.summary = summary_line(&rep, rel_l2);
    let worst = r.cap(
        rep["worst"].as_array().cloned().unwrap_or_default(),
        max_items(args),
    );
    let d = &mut r.data;
    d.insert("verdict".into(), rep["verdict"].clone());
    d.insert("summary".into(), rep["summary"].clone());
    d.insert("threshold_rel_l2".into(), num(rel_l2));
    d.insert("a".into(), rep["a"]["source"].clone());
    d.insert("b".into(), rep["b"]["source"].clone());
    d.insert("worst".into(), worst);
    Ok(r)
}

fn weights(args: &Map<String, Value>, argv: &[String]) -> Result<Envelope, CumlxError> {
    let py = python_command(argv);
    if args["loader"].as_str() != Some("mlx_lm") {
        return Err(python_only(
            "--loader other than mlx_lm runs Python code; cumlx-rs reads the port files only",
            &py,
        ));
    }
    if !args["mapping"].is_null() {
        return Err(python_only(
            "--mapping applies Python regular expressions; cumlx-rs does not",
            &py,
        ));
    }
    let output = args["output"].as_str();
    let overwrite = args["overwrite"].as_bool().unwrap_or(false);
    if let Some(o) = output {
        if Path::new(o).exists() && !overwrite {
            return Err(CumlxError::new("E_OUTPUT_EXISTS", format!("{o} exists"))
                .hint("pass --overwrite or choose a new path"));
        }
        if Path::new(o).is_dir() {
            return Err(
                CumlxError::new("E_OUTPUT_EXISTS", format!("{o} is a directory"))
                    .hint("choose a file path for -o/--output"),
            );
        }
    }
    let (original, port) = (
        args["original"].as_str().unwrap(),
        args["port"].as_str().unwrap(),
    );
    log(
        args,
        &format!("comparing the weights of {port} with {original}"),
    );
    let rep = run_weights(original, port, &py)?;
    let mut r = Envelope::new("weights");
    let verdict = rep["verdict"].as_str().unwrap();
    r.exit_code = ExitCode::from_verdict(verdict);
    let s = rep["summary"].as_object().unwrap();
    let first = &rep["first_failure"];
    let mut summary = format!(
        "{verdict}: {} pass, {} fail, {} unverified of {} parameters",
        s["pass"], s["fail"], s["unverified"], s["parameters"]
    );
    if let Some(f) = first.as_object() {
        summary.push_str(&format!(
            "; first failure `{}`",
            f["name"].as_str().unwrap_or("")
        ));
        if let Some(d) = f
            .get("diagnosis")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
        {
            summary.push_str(&format!(" ({d})"));
        }
    }
    r.summary = summary;
    let rows = rep["parameters"].as_array().unwrap();
    let pick = |status: &str| {
        rows.iter()
            .filter(|x| x["status"] == status)
            .cloned()
            .collect::<Vec<_>>()
    };
    let failures = r.cap(pick("fail"), max_items(args));
    let unverified = r.cap(pick("unverified"), max_items(args));
    let extra = r.cap(
        rep["extra_in_port"].as_array().cloned().unwrap_or_default(),
        max_items(args),
    );
    let d = &mut r.data;
    d.insert("verdict".into(), verdict.into());
    d.insert("summary".into(), rep["summary"].clone());
    d.insert("first_failure".into(), first.clone());
    d.insert("failures".into(), failures);
    d.insert("unverified".into(), unverified);
    d.insert("extra_in_port".into(), extra);
    if let Some(o) = output {
        let text = json_dumps(
            &Value::Object(rep.clone()),
            Dumps {
                indent: Some(1),
                ensure_ascii: true,
            },
        ) + "\n";
        std::fs::write(o, text).map_err(|e| {
            CumlxError::new("E_USAGE", format!("cannot write {o}: {e}"))
                .hint("choose a writable -o/--output path")
        })?;
        r.artifact("weights_report", o);
    }
    if verdict == "ok" {
        r.suggest(
            &format!(
                "cumlx capture {} --prompt Hello -o golden/ && cumlx diff golden/ {}",
                shlex_quote(original),
                shlex_quote(port)
            ),
            "the weights are right; now check the code of the port",
        );
    }
    Ok(r)
}

fn refuse(cmd: &spec::Command, argv: &[String]) -> CumlxError {
    let what = if cmd.needs.starts_with("nothing") {
        format!(
            "`cumlx {}` is not ported to Rust; only the Python cumlx runs it",
            cmd.name
        )
    } else {
        format!(
            "`cumlx {}` needs {}, which only the Python cumlx has; cumlx-rs does not run it",
            cmd.name, cmd.needs
        )
    };
    let mut install = cmd.install.to_string();
    if cmd.name == "kernel capture" {
        install.push_str(" (on an NVIDIA machine with nvcc)");
    }
    CumlxError::new("E_PYTHON_ONLY", what)
        .hint(format!("{install} && {}", python_command(argv)))
        .detail("command", cmd.name)
        .detail(
            "requires",
            cmd.requires
                .iter()
                .map(|r| Value::from(*r))
                .collect::<Vec<_>>(),
        )
}

fn dispatch(argv: &[String]) -> Result<Envelope, CumlxError> {
    let name = command_name(argv);
    let all = spec::commands();
    let first = argv.first().map(String::as_str).unwrap_or("");
    let (cmd, rest) = if first == "kernel" {
        let sub = format!("kernel {}", argv.get(1).map(String::as_str).unwrap_or(""));
        match all.iter().find(|c| c.name == sub) {
            Some(c) => (c, &argv[2..]),
            None => {
                return Ok(usage(
                    &name,
                    "argument kernel needs a subcommand: capture, verify or port".into(),
                ));
            }
        }
    } else {
        match all.iter().find(|c| c.name == first) {
            Some(c) => (c, &argv[1..]),
            None => {
                let msg = if first.starts_with('-') {
                    format!("unrecognized arguments before the command: {first}")
                } else {
                    format!("argument COMMAND: invalid choice: '{first}'")
                };
                return Ok(usage(&name, msg));
            }
        }
    };
    if cmd.args.is_none() {
        return Err(refuse(cmd, argv));
    }
    let parsed = match args::parse(cmd, rest) {
        Ok(p) => p,
        Err(args::Usage(m)) => return Ok(usage(cmd.name, m)),
    };
    match cmd.name {
        "describe" => describe(&parsed),
        "tracediff" => tracediff(&parsed),
        "weights" => weights(&parsed, argv),
        other => unreachable!("no handler for {other}"),
    }
}

fn help_text() -> String {
    let mut lines = vec![
        format!("usage: {TOOL} COMMAND [options]"),
        String::new(),
        "The Rust core of CUMLX. Agents: run `cumlx-rs describe`.".into(),
        String::new(),
        "commands run here:".into(),
    ];
    let all = spec::commands();
    for c in all.iter().filter(|c| c.args.is_some()) {
        lines.push(format!("  {:<15} {}", c.name, c.summary));
    }
    lines.push(String::new());
    lines.push("commands of the Python cumlx (they exit 3 with E_PYTHON_ONLY here):".into());
    let names: Vec<&str> = all
        .iter()
        .filter(|c| c.args.is_none())
        .map(|c| c.name)
        .collect();
    lines.push(format!("  {}", names.join(", ")));
    lines.join("\n")
}

fn run(mut argv: Vec<String>) -> i32 {
    let fmt = detect_format(&argv);
    let started = Instant::now();
    let first_is_option = argv.first().is_some_and(|a| {
        a.starts_with('-') && !["-h", "--help", "--version"].contains(&a.as_str())
    });
    if argv.is_empty() || (first_is_option && command_name(&argv) == TOOL) {
        argv.insert(0, "describe".into()); // a bare `cumlx-rs`, optionally with output flags, describes itself
    }
    let mut stdout = std::io::stdout();
    if argv.iter().any(|a| a == "--version") {
        let _ = writeln!(stdout, "{TOOL} {}", cumlx::VERSION);
        return 0;
    }
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        let _ = writeln!(stdout, "{}", help_text());
        return 0;
    }
    let name = command_name(&argv);
    let outcome = std::panic::catch_unwind(|| dispatch(&argv));
    let result = match outcome {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => Envelope::from_error(&name, &e),
        Err(panic) => {
            let text = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            let mut r = Envelope::new(&name);
            r.exit_code = ExitCode::Internal;
            r.summary = format!("internal error: {text}");
            r.errors.push(
                CumlxError::new("E_INTERNAL", text)
                    .hint("this is a CUMLX bug; include this JSON in the report")
                    .to_value(),
            );
            r
        }
    };
    let doc = result.to_value(cumlx::VERSION, started.elapsed().as_millis() as u64);
    let text = if fmt == "json" {
        to_json(&doc)
    } else {
        to_text(&doc, TOOL)
    };
    let _ = stdout.write_all(text.as_bytes());
    let _ = stdout.write_all(b"\n");
    let _ = stdout.flush();
    result.exit_code.code()
}

fn main() {
    let argv: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    std::process::exit(run(argv));
}
