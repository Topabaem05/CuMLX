//! Text formatting that matches the Python implementation byte for byte.
//!
//! The Python `cumlx` writes floats with `repr()`, `format(x, 'g')` and `json.dumps`. An agent that reads the
//! output of both implementations must see the same text, so these functions follow CPython's rules:
//! shortest round-trip digits, the switch to exponent notation, two-digit exponents and JSON separators.

use serde_json::{Map, Value};

/// `repr(x)` of a Python float: shortest digits that read back to `x`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let (negative, digits, exp) = shortest_digits(x);
    let decpt = exp + 1;
    let body = if decpt <= -4 || decpt > 16 {
        let mut m = digits[..1].to_string();
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        format!(
            "{m}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    if negative {
        format!("-{body}")
    } else {
        body
    }
}

/// Sign, decimal digits without a point, and the exponent of the first digit.
fn shortest_digits(x: f64) -> (bool, String, i32) {
    split_exponent(&format!("{x:e}"))
}

fn split_exponent(s: &str) -> (bool, String, i32) {
    let (mant, exp) = s.split_once('e').expect("exponent form");
    let negative = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    (negative, digits, exp.parse().expect("exponent"))
}

/// `format(x, f'.{precision}g')` of a Python float. `format(x, 'g')` is precision 6.
pub fn format_g(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let p = precision.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let (negative, digits, exp) = split_exponent(&format!("{:.*e}", p - 1, x));
    let body = if exp >= -4 && exp < p as i32 {
        let fixed = format!("{:.*}", (p as i32 - 1 - exp) as usize, x.abs());
        strip_fraction_zeros(&fixed)
    } else {
        let mut m = digits[..1].to_string();
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        format!(
            "{}e{}{:02}",
            strip_fraction_zeros(&m),
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
    };
    if negative {
        format!("-{body}")
    } else {
        body
    }
}

fn strip_fraction_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// `str(value)` of a Python object that came from JSON, as an f-string writes it.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => number_text(n),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => py_repr(v),
    }
}

/// `repr(value)` of a Python object that came from JSON.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            format!(
                "[{}]",
                items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
            )
        }
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, x)| format!("{}: {}", repr_str(k), py_repr(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => py_str(other),
    }
}

/// `repr(s)` of a Python string.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c if (0x80..0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `shlex.quote(s)`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

/// `str(pathlib.PurePosixPath(s))`: repeated and trailing slashes and `.` parts removed.
pub fn py_path(s: &str) -> String {
    if s.is_empty() {
        return ".".into();
    }
    let root = if s.starts_with("//") && !s.starts_with("///") {
        "//"
    } else if s.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = s
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let joined = parts.join("/");
    if root.is_empty() && joined.is_empty() {
        ".".into()
    } else {
        format!("{root}{joined}")
    }
}

/// `str(PurePosixPath(a) / b)`.
pub fn py_path_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        py_path(b)
    } else {
        let base = py_path(a);
        if base == "." {
            py_path(b)
        } else {
            py_path(&format!("{base}/{b}"))
        }
    }
}

fn number_text(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        float_repr(n.as_f64().unwrap_or(f64::NAN))
    }
}

/// Options of `json.dumps`.
#[derive(Clone, Copy, Debug)]
pub struct Dumps {
    /// `indent=`; `None` writes one line with `, ` and `: ` separators.
    pub indent: Option<usize>,
    /// `ensure_ascii=`: escape every character above U+007F.
    pub ensure_ascii: bool,
}

/// `json.dumps(value, indent=..., ensure_ascii=...)`. Non-finite floats cannot occur in a `Value`.
pub fn json_dumps(v: &Value, opts: Dumps) -> String {
    let mut out = String::new();
    write_value(&mut out, v, opts, 0);
    out
}

fn write_value(out: &mut String, v: &Value, opts: Dumps, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_text(n)),
        Value::String(s) => write_string(out, s, opts.ensure_ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                separator(out, i, opts, level + 1);
                write_value(out, item, opts, level + 1);
            }
            close(out, opts, level);
            out.push(']');
        }
        Value::Object(map) => write_object(out, map, opts, level),
    }
}

fn write_object(out: &mut String, map: &Map<String, Value>, opts: Dumps, level: usize) {
    if map.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push('{');
    for (i, (k, item)) in map.iter().enumerate() {
        separator(out, i, opts, level + 1);
        write_string(out, k, opts.ensure_ascii);
        out.push_str(": ");
        write_value(out, item, opts, level + 1);
    }
    close(out, opts, level);
    out.push('}');
}

fn separator(out: &mut String, index: usize, opts: Dumps, level: usize) {
    match opts.indent {
        Some(n) => {
            if index > 0 {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
        None => {
            if index > 0 {
                out.push_str(", ");
            }
        }
    }
}

fn close(out: &mut String, opts: Dumps, level: usize) {
    if let Some(n) = opts.indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_follows_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789.125, "123456789.125"),
            (-2.5e-300, "-2.5e-300"),
            (1e100, "1e+100"),
            (0.30000000000000004, "0.30000000000000004"),
            (f64::NAN, "nan"),
            (-0.0, "-0.0"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn format_g_follows_python() {
        let cases = [
            (1e-4, 6, "0.0001"),
            (1e-5, 6, "1e-05"),
            (2.0, 6, "2"),
            (1.5, 6, "1.5"),
            (123456.0, 6, "123456"),
            (1234567.0, 6, "1.23457e+06"),
            (0.000123456789, 6, "0.000123457"),
            (2.5, 1, "2"),
            (0.5, 1, "0.5"),
            (1.0000001, 6, "1"),
            (-3.25e-7, 6, "-3.25e-07"),
            (0.0, 6, "0"),
        ];
        for (x, p, want) in cases {
            assert_eq!(format_g(x, p), want, "{x} .{p}g");
        }
    }

    #[test]
    fn json_dumps_follows_python() {
        let v = json!({"a": [1, 2.0, "x\u{e9}"], "b": {}, "c": [], "d": null, "e": "q\"\n"});
        assert_eq!(
            json_dumps(&v, Dumps { indent: None, ensure_ascii: false }),
            "{\"a\": [1, 2.0, \"x\u{e9}\"], \"b\": {}, \"c\": [], \"d\": null, \"e\": \"q\\\"\\n\"}"
        );
        assert_eq!(
            json_dumps(
                &json!({"a": [1, "\u{e9}\u{1F600}"]}),
                Dumps {
                    indent: Some(1),
                    ensure_ascii: true
                }
            ),
            "{\n \"a\": [\n  1,\n  \"\\u00e9\\ud83d\\ude00\"\n ]\n}"
        );
    }

    #[test]
    fn paths_and_quotes_follow_python() {
        assert_eq!(py_path("golden/"), "golden");
        assert_eq!(py_path("./a//b/./c/"), "a/b/c");
        assert_eq!(py_path("/x/"), "/x");
        assert_eq!(
            py_path_join("g/", "samples/s0000.safetensors"),
            "g/samples/s0000.safetensors"
        );
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("./ok-1.x"), "./ok-1.x");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a"), "'a'");
    }
}
