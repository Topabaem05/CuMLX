//! An argument parser that accepts what Python's argparse accepts for these commands: options before or after
//! positionals, `--opt=value`, `-ovalue`, unique prefixes of long options, `--` to end options, and a negative
//! number such as `-1` as a value.

use crate::spec::{Arg, Command, Kind, Ty};
use serde_json::{Map, Value};

/// A usage error, reported as `E_USAGE`.
#[derive(Debug, Clone, PartialEq)]
pub struct Usage(pub String);

fn negative_number(s: &str) -> bool {
    // argparse: '^-\d+$|^-\d*\.\d+$'
    let Some(rest) = s.strip_prefix('-') else {
        return false;
    };
    if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    match rest.split_once('.') {
        Some((a, b)) => {
            a.chars().all(|c| c.is_ascii_digit())
                && !b.is_empty()
                && b.chars().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}

fn looks_like_option(s: &str) -> bool {
    s.starts_with('-') && s != "-" && !negative_number(s)
}

fn shown(arg: &Arg) -> String {
    if arg.positional() {
        arg.name.to_string()
    } else {
        arg.flags.join("/")
    }
}

fn convert(arg: &Arg, raw: &str) -> Result<Value, Usage> {
    let v: Value = match arg.ty {
        Ty::Str => raw.into(),
        Ty::Int => raw
            .trim()
            .replace('_', "")
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| {
                Usage(format!(
                    "argument {}: invalid int value: '{raw}'",
                    shown(arg)
                ))
            })?,
        Ty::Float => {
            let x = raw.trim().replace('_', "").parse::<f64>().map_err(|_| {
                Usage(format!(
                    "argument {}: invalid float value: '{raw}'",
                    shown(arg)
                ))
            })?;
            // Keep infinities and NaN, which `float()` accepts and a JSON number cannot hold.
            float_value(x)
        }
        Ty::Bool => true.into(),
    };
    if let (Some(choices), Some(s)) = (&arg.choices, v.as_str()) {
        if !choices.iter().any(|c| c == s) {
            let list = choices
                .iter()
                .map(|c| format!("'{c}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Usage(format!(
                "argument {}: invalid choice: '{s}' (choose from {list})",
                shown(arg)
            )));
        }
    }
    Ok(v)
}

/// A float that may be infinite or NaN, kept as text when JSON cannot hold it.
fn float_value(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or_else(|| Value::String(format!("{x}")))
}

/// Read a float argument back, including `inf` and `NaN`.
pub fn get_float(args: &Map<String, Value>, name: &str) -> f64 {
    match &args[name] {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => s.parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

/// Parse the arguments of one command. Defaults fill what is not given.
pub fn parse(cmd: &Command, argv: &[String]) -> Result<Map<String, Value>, Usage> {
    let spec = cmd.args.as_ref().expect("an implemented command");
    let mut out = Map::new();
    let mut positionals: Vec<&String> = Vec::new();
    let mut unrecognized: Vec<String> = Vec::new();
    let mut only_positional = false;
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        i += 1;
        if only_positional || !looks_like_option(a) {
            positionals.push(a);
            continue;
        }
        if a == "--" {
            only_positional = true;
            continue;
        }
        let (flag, inline) = if a.starts_with("--") {
            match a.split_once('=') {
                Some((f, v)) => (f.to_string(), Some(v.to_string())),
                None => (a.clone(), None),
            }
        } else {
            let (f, rest) = a.split_at(2.min(a.len()));
            (
                f.to_string(),
                (!rest.is_empty()).then(|| rest.trim_start_matches('=').to_string()),
            )
        };
        let exact = spec.iter().find(|x| x.flags.contains(&flag.as_str()));
        let arg = match exact {
            Some(x) => x,
            None if flag.starts_with("--") => {
                let matches: Vec<&Arg> = spec
                    .iter()
                    .filter(|x| {
                        x.flags
                            .iter()
                            .any(|f| f.starts_with("--") && f.starts_with(&flag))
                    })
                    .collect();
                match matches.len() {
                    1 => matches[0],
                    0 => {
                        unrecognized.push(a.clone());
                        continue;
                    }
                    _ => {
                        let names = matches
                            .iter()
                            .flat_map(|x| x.flags.iter().filter(|f| f.starts_with(&flag)))
                            .copied()
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(Usage(format!(
                            "ambiguous option: {flag} could match {names}"
                        )));
                    }
                }
            }
            None => {
                unrecognized.push(a.clone());
                continue;
            }
        };
        match arg.kind {
            Kind::Flag => {
                if let Some(v) = inline {
                    return Err(Usage(format!(
                        "argument {}: ignored explicit argument '{v}'",
                        shown(arg)
                    )));
                }
                out.insert(arg.name.into(), true.into());
            }
            Kind::Value => {
                let raw = match inline {
                    Some(v) => v,
                    None => match argv.get(i) {
                        Some(next) if !looks_like_option(next) => {
                            i += 1;
                            next.clone()
                        }
                        _ => {
                            return Err(Usage(format!(
                                "argument {}: expected one argument",
                                shown(arg)
                            )))
                        }
                    },
                };
                out.insert(arg.name.into(), convert(arg, &raw)?);
            }
        }
    }
    let wanted: Vec<&Arg> = spec.iter().filter(|x| x.positional()).collect();
    if positionals.len() < wanted.len() {
        let missing: Vec<&str> = wanted[positionals.len()..].iter().map(|x| x.name).collect();
        return Err(Usage(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    for (arg, raw) in wanted.iter().zip(&positionals) {
        out.insert(arg.name.into(), convert(arg, raw)?);
    }
    unrecognized.extend(positionals[wanted.len()..].iter().map(|s| s.to_string()));
    if !unrecognized.is_empty() {
        return Err(Usage(format!(
            "unrecognized arguments: {}",
            unrecognized.join(" ")
        )));
    }
    for arg in spec {
        if !out.contains_key(arg.name) {
            out.insert(arg.name.into(), arg.default.clone());
        }
    }
    if !out["max_items"].is_i64() {
        return Err(Usage("CUMLX_MAX_ITEMS must be an integer".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::commands;

    fn cmd(name: &str) -> Command {
        commands().into_iter().find(|c| c.name == name).unwrap()
    }

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn options_go_anywhere() {
        let a = parse(
            &cmd("tracediff"),
            &argv(&["--rel-l2=1e-3", "x", "--max-items", "-1", "y", "-v"]),
        )
        .unwrap();
        assert_eq!((a["a"].as_str(), a["b"].as_str()), (Some("x"), Some("y")));
        assert_eq!(
            (
                get_float(&a, "rel_l2"),
                a["max_items"].as_i64(),
                a["verbose"].as_bool()
            ),
            (1e-3, Some(-1), Some(true))
        );
        let w = parse(&cmd("weights"), &argv(&["o", "p", "-ow.json", "--over"])).unwrap();
        assert_eq!(
            (
                w["output"].as_str(),
                w["overwrite"].as_bool(),
                w["loader"].as_str()
            ),
            (Some("w.json"), Some(true), Some("mlx_lm"))
        );
        assert_eq!(
            get_float(
                &parse(&cmd("tracediff"), &argv(&["x", "y", "--rel-l2", "inf"])).unwrap(),
                "rel_l2"
            ),
            f64::INFINITY
        );
    }

    #[test]
    fn mistakes_are_usage_errors() {
        let t = cmd("tracediff");
        assert!(parse(&t, &argv(&["x"]))
            .unwrap_err()
            .0
            .contains("required: b"));
        assert!(parse(&t, &argv(&["x", "y", "z"]))
            .unwrap_err()
            .0
            .contains("unrecognized arguments: z"));
        assert!(parse(&t, &argv(&["x", "y", "--rel-l2", "abc"]))
            .unwrap_err()
            .0
            .contains("invalid float"));
        assert!(parse(&t, &argv(&["x", "y", "--format", "xml"]))
            .unwrap_err()
            .0
            .contains("invalid choice"));
        assert!(parse(&t, &argv(&["x", "y", "--rel-l2"]))
            .unwrap_err()
            .0
            .contains("expected one argument"));
        assert!(parse(&t, &argv(&["x", "y", "--nope"]))
            .unwrap_err()
            .0
            .contains("unrecognized"));
    }
}
