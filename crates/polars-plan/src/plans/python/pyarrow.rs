use std::fmt::Write;

use polars_core::datatypes::AnyValue;
#[cfg(feature = "dtype-datetime")]
use polars_core::prelude::TimeZone;
use polars_core::prelude::{DataType, ExplodeOptions, Schema, TimeUnit};
use polars_core::series::Series;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
use pyo3::types::PyDate;
use pyo3::types::PyList;
#[cfg(feature = "dtype-datetime")]
use pyo3::types::{PyDateTime, PyTzInfo};

use crate::prelude::*;

// Don't convert more than this amount of items to Python objects.
const LIST_ITEM_LIMIT: usize = 100;

#[cfg(feature = "is_in")]
pub(crate) enum IsInHaystack {
    Empty, // fast path for when haystack is empty; returns False
    Series(Series),
}

#[cfg(feature = "is_in")]
pub(crate) fn needle_isin_haystack(lv: &LiteralValue, nulls_equal: bool) -> Option<IsInHaystack> {
    if !lv.get_datatype().is_list() {
        return None;
    }

    let mut haystack_series = if let LiteralValue::Series(s) = lv
        && s.dtype().is_list()
        && s.len() == 1
    {
        if s.null_count() == 0 {
            s.explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            })
            .ok()?
        } else {
            Series::full_null(PlSmallStr::EMPTY, 0, &DataType::Null)
        }
    } else if let Some(AnyValue::List(s)) = lv.to_any_value() {
        s
    } else if lv.is_null() {
        Series::full_null(PlSmallStr::EMPTY, 0, &DataType::Null)
    } else {
        return None;
    };

    let converted_len = haystack_series.len()
        - if nulls_equal {
            0
        } else {
            haystack_series.null_count()
        };

    if converted_len > LIST_ITEM_LIMIT {
        return None;
    }
    if converted_len == 0 {
        return Some(IsInHaystack::Empty);
    }
    if !nulls_equal {
        haystack_series = haystack_series.drop_nulls();
    }
    Some(IsInHaystack::Series(haystack_series))
}

#[cfg(feature = "dtype-datetime")]
fn to_py_datetime(v: i64, tu: &TimeUnit, tz: Option<&TimeZone>) -> String {
    // note: `to_py_datetime` and the `Datetime`
    // dtype have to be in-scope on the python side
    match tz {
        None => format!("to_py_datetime({},'{}')", v, tu.to_ascii()),
        Some(tz) => format!("to_py_datetime({},'{}','{}')", v, tu.to_ascii(), tz),
    }
}

fn sanitize(name: &str) -> Option<&str> {
    if name.chars().all(|c| match c {
        ' ' => true,
        '-' => true,
        '_' => true,
        c => c.is_alphanumeric(),
    }) {
        Some(name)
    } else {
        None
    }
}

/// Render a flat `Series` as a Python list literal, e.g. `[1,2,3]`.
///
/// Returns `None` for values we cannot faithfully (or safely) write out as
/// source text.
fn series_to_pyarrow_list(s: &Series) -> Option<String> {
    let mut list_repr = String::with_capacity(s.len() * 5);
    list_repr.push('[');
    for av in s.iter() {
        match av {
            AnyValue::Null => list_repr.push_str("None,"),
            AnyValue::Boolean(v) => {
                let s = if v { "True" } else { "False" };
                write!(list_repr, "{s},").unwrap();
            },
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, tz) => {
                let dtm = to_py_datetime(v, &tu, tz);
                write!(list_repr, "{dtm},").unwrap();
            },
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => {
                write!(list_repr, "to_py_date({v}),").unwrap();
            },
            AnyValue::String(s) => {
                let _ = sanitize(s)?;
                write!(list_repr, "{av},").unwrap();
            },
            // Hard to sanitize
            AnyValue::Binary(_) | AnyValue::List(_) => return None,
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(_, _) => return None,
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(_, _, _) => return None,
            _ => {
                write!(list_repr, "{av},").unwrap();
            },
        }
    }
    // pop last comma
    list_repr.pop();
    list_repr.push(']');
    Some(list_repr)
}

/// Shared decision tree for lowering an `AExpr` predicate to either a
/// PyArrow-expression source string (consumed by Iceberg's AST walker and by
/// Delta's `eval`) or a live PyArrow `Expression` object (consumed directly
/// by `scan_pyarrow_dataset`).
///
/// This exists so the two renderers below cannot silently diverge on what
/// they consider valid to push down: every *tree-shape* decision (which
/// `AExpr` variants translate, how `is_between`/`is_in`/`starts_with`/
/// validity-comparisons restructure, which arithmetic is safe) is made
/// exactly once, here, rather than duplicated across two ~500-line matches.
///
/// Literal encoding is deliberately *not* unified: writing source text and
/// building a live object are irreducibly different tasks, so each renderer
/// encodes a `LiteralValue` independently.
///
/// Not every variant is supported by every renderer. `scan_pyarrow_dataset`
/// has never covered `ValidityCompare` (`eq_missing`/`ne_missing`) or
/// `StartsWith` — there's simply been no need to, since those only matter to
/// Iceberg/Delta. Renderers return `None` for a variant they don't support,
/// preserving that pre-existing asymmetry rather than papering over it.
enum PaIr {
    /// Raw (unsanitized) column name — sanitization is a string-representation
    /// concern only (the value gets parsed as source text); the live-object
    /// path never needed it, so it isn't applied here.
    Field(PlSmallStr),
    Literal(LiteralValue),
    BinOp {
        left: Box<PaIr>,
        op: Operator,
        right: Box<PaIr>,
    },
    Xor(Box<PaIr>, Box<PaIr>),
    /// `eq_missing`/`ne_missing`. `literal: None` is a null-literal comparison
    /// (`is_null`/`~is_null`); `Some` is the compound null-safe-equality form.
    ValidityCompare {
        column: Box<PaIr>,
        literal: Option<Box<PaIr>>,
        eq: bool,
    },
    Not(Box<PaIr>),
    IsNull(Box<PaIr>),
    IsNotNull(Box<PaIr>),
    /// The `bool` is whether the operand's dtype supports `is_nan` at all
    /// (`false` for e.g. `Decimal`). Only the object renderer enforces this
    /// today, matching `aexpr_to_pyarrow`'s existing guard; the string
    /// renderer doesn't check it, matching `predicate_to_pa`'s current
    /// behavior. Computed once here either way, since it needs the schema.
    IsNan(Box<PaIr>, bool),
    IsNotNan(Box<PaIr>, bool),
    StartsWith(Box<PaIr>, String),
    #[cfg(feature = "is_in")]
    IsIn(Box<PaIr>, IsInHaystack),
    Between {
        column: Box<PaIr>,
        left_op: Operator,
        lower: Box<PaIr>,
        right_op: Operator,
        upper: Box<PaIr>,
    },
}

/// Build the shared IR for `predicate`. `schema` is the scan output schema,
/// used to resolve column dtypes so that only arithmetic provably equivalent
/// to Polars' is lowered (see [`is_float64_arithmetic`]), and so the
/// `is_nan`/`is_not_nan` dtype guard can be evaluated.
fn to_pa_ir(predicate: Node, expr_arena: &Arena<AExpr>, schema: &Schema) -> Option<PaIr> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => match op {
            Operator::EqValidity | Operator::NotEqValidity => {
                // The column is repeated in the output, so restrict this to plain columns.
                let (column, literal_node, lv) =
                    match (expr_arena.get(*left), expr_arena.get(*right)) {
                        (AExpr::Column(_), AExpr::Literal(lv)) => (*left, *right, lv),
                        (AExpr::Literal(lv), AExpr::Column(_)) => (*right, *left, lv),
                        _ => return None,
                    };

                let eq = matches!(op, Operator::EqValidity);
                let column = Box::new(to_pa_ir(column, expr_arena, schema)?);

                let literal = if lv.is_null() {
                    None
                } else {
                    Some(Box::new(to_pa_ir(literal_node, expr_arena, schema)?))
                };

                Some(PaIr::ValidityCompare {
                    column,
                    literal,
                    eq,
                })
            },
            Operator::Xor => {
                if !(returns_boolean(*left, expr_arena) && returns_boolean(*right, expr_arena)) {
                    return None;
                }

                let l = to_pa_ir(*left, expr_arena, schema)?;
                let r = to_pa_ir(*right, expr_arena, schema)?;
                Some(PaIr::Xor(Box::new(l), Box::new(r)))
            },
            op => {
                reject_inexact_arithmetic(*left, *right, *op, expr_arena, schema)?;

                let left = to_pa_ir(*left, expr_arena, schema)?;
                let right = to_pa_ir(*right, expr_arena, schema)?;

                Some(PaIr::BinOp {
                    left: Box::new(left),
                    op: *op,
                    right: Box::new(right),
                })
            },
        },
        AExpr::Column(name) => Some(PaIr::Field(name.clone())),
        // Only meaningful as the haystack of an `is_in`, which handles it itself.
        AExpr::Literal(LiteralValue::Series(_)) => None,
        AExpr::Literal(lv) => Some(PaIr::Literal(lv.clone())),
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal }),
            input,
            ..
        } => {
            let col = to_pa_ir(input.first()?.node(), expr_arena, schema)?;

            let AExpr::Literal(lv) = expr_arena.get(input.get(1)?.node()) else {
                return None;
            };

            let haystack = needle_isin_haystack(lv, *nulls_equal)?;
            Some(PaIr::IsIn(Box::new(col), haystack))
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            if !matches!(expr_arena.get(input.first()?.node()), AExpr::Column(_)) {
                return None;
            }

            let column = to_pa_ir(input.first()?.node(), expr_arena, schema)?;
            let left_op = match closed {
                ClosedInterval::None | ClosedInterval::Right => Operator::Gt,
                ClosedInterval::Both | ClosedInterval::Left => Operator::GtEq,
            };
            let right_op = match closed {
                ClosedInterval::None | ClosedInterval::Left => Operator::Lt,
                ClosedInterval::Both | ClosedInterval::Right => Operator::LtEq,
            };

            let lower = to_pa_ir(input.get(1)?.node(), expr_arena, schema)?;
            let upper = to_pa_ir(input.get(2)?.node(), expr_arena, schema)?;

            Some(PaIr::Between {
                column: Box::new(column),
                left_op,
                lower: Box::new(lower),
                right_op,
                upper: Box::new(upper),
            })
        },
        #[cfg(feature = "strings")]
        AExpr::Function {
            function: IRFunctionExpr::StringExpr(IRStringFunction::StartsWith),
            input,
            ..
        } => {
            let col = to_pa_ir(input.first()?.node(), expr_arena, schema)?;
            let AExpr::Literal(lv) = expr_arena.get(input.get(1)?.node()) else {
                return None;
            };
            let prefix = sanitize(lv.extract_str()?)?.to_string();
            Some(PaIr::StartsWith(Box::new(col), prefix))
        },
        AExpr::Function {
            function, input, ..
        } => {
            let input_expr = input.first()?;
            let input_ir = to_pa_ir(input_expr.node(), expr_arena, schema)?;

            match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::Not) => {
                    Some(PaIr::Not(Box::new(input_ir)))
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                    Some(PaIr::IsNull(Box::new(input_ir)))
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => {
                    Some(PaIr::IsNotNull(Box::new(input_ir)))
                },
                // note: only applies to primitive (non-decimal) numeric types
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNan) => {
                    let dtype = input_expr.dtype(schema, expr_arena).ok()?;
                    let valid = dtype.is_primitive_numeric() || dtype.is_null();
                    Some(PaIr::IsNan(Box::new(input_ir), valid))
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNan) => {
                    let dtype = input_expr.dtype(schema, expr_arena).ok()?;
                    let valid = dtype.is_primitive_numeric() || dtype.is_null();
                    Some(PaIr::IsNotNan(Box::new(input_ir), valid))
                },
                _ => None,
            }
        },
        _ => None,
    }
}

// Build an eval-able / AST-walker-compatible string predicate (e.g.
// `pa.compute.field('x') > pa.compute.scalar(1)`). Used by the iceberg and
// delta paths which feed the string into Python (delta `eval`s it,
// iceberg walks it via `try_convert_pyarrow_predicate`).
//
// `schema` is the scan output schema, used to resolve column dtypes so that
// only arithmetic provably equivalent to Polars' is lowered (see
// [`is_float64_arithmetic`]).
pub fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<String> {
    render_pa_string(&to_pa_ir(predicate, expr_arena, schema)?)
}

fn render_pa_string(ir: &PaIr) -> Option<String> {
    match ir {
        PaIr::Field(name) => {
            let name = sanitize(name)?;
            Some(format!("pa.compute.field('{name}')"))
        },
        PaIr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let dtype = av.dtype();
            match av.as_borrowed() {
                AnyValue::String(s) => {
                    let s = sanitize(s)?;
                    Some(format!("'{s}'"))
                },
                AnyValue::Boolean(val) => {
                    if val {
                        Some("pa.compute.scalar(True)".to_string())
                    } else {
                        Some("pa.compute.scalar(False)".to_string())
                    }
                },
                #[cfg(feature = "dtype-date")]
                AnyValue::Date(v) => Some(format!("to_py_date({v})")),
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz)),
                AnyValue::Binary(_) | AnyValue::List(_) => None,
                #[cfg(feature = "dtype-array")]
                AnyValue::Array(_, _) => None,
                #[cfg(feature = "dtype-struct")]
                AnyValue::Struct(_, _, _) => None,
                av => {
                    if dtype.is_float() {
                        let val = av.extract::<f64>()?;
                        Some(format!("{val}"))
                    } else if dtype.is_integer() {
                        let val = av.extract::<i64>()?;
                        Some(format!("{val}"))
                    } else {
                        None
                    }
                },
            }
        },
        PaIr::BinOp { left, op, right } => {
            let symbol = binary_op_symbol(op)?;
            let mut lhs = render_pa_string(left)?;
            let rhs = render_pa_string(right)?;

            if op.is_arithmetic() {
                // PyArrow expressions define no reflected arithmetic operators, so a
                // bare Python literal on the left raises `TypeError` instead of
                // building an expression. (Comparisons are fine: Python falls back
                // to the reflected comparison on the right-hand expression.)
                if matches!(**left, PaIr::Literal(_)) && !lhs.starts_with("pa.compute.") {
                    lhs = format!("pa.compute.scalar({lhs})");
                }
            }

            Some(format!("({lhs} {symbol} {rhs})"))
        },
        PaIr::Xor(l, r) => {
            let lhs = render_pa_string(l)?;
            let rhs = render_pa_string(r)?;
            Some(format!("(({lhs} | {rhs}) & ~({lhs} & {rhs}))"))
        },
        PaIr::ValidityCompare {
            column,
            literal,
            eq,
        } => {
            let column = render_pa_string(column)?;

            Some(match literal {
                None => {
                    if *eq {
                        format!("({column}).is_null()")
                    } else {
                        format!("~({column}).is_null()")
                    }
                },
                Some(literal) => {
                    let literal = render_pa_string(literal)?;

                    // A null column value is not equal to a non-null literal, whereas
                    // the plain comparison would evaluate to null.
                    if *eq {
                        format!("(({column} == {literal}) & ~({column}).is_null())")
                    } else {
                        format!("(({column} != {literal}) | ({column}).is_null())")
                    }
                },
            })
        },
        PaIr::Not(inner) => Some(format!("~({})", render_pa_string(inner)?)),
        PaIr::IsNull(inner) => Some(format!("({}).is_null()", render_pa_string(inner)?)),
        PaIr::IsNotNull(inner) => Some(format!("~({}).is_null()", render_pa_string(inner)?)),
        PaIr::IsNan(inner, _valid) => Some(format!("({}).is_nan()", render_pa_string(inner)?)),
        PaIr::IsNotNan(inner, _valid) => Some(format!("~({}).is_nan()", render_pa_string(inner)?)),
        PaIr::StartsWith(col, prefix) => {
            let col = render_pa_string(col)?;
            Some(format!("pa.compute.starts_with({col}, pattern='{prefix}')"))
        },
        #[cfg(feature = "is_in")]
        PaIr::IsIn(col, haystack) => {
            let col = render_pa_string(col)?;
            match haystack {
                IsInHaystack::Empty => Some("pa.compute.scalar(False)".to_string()),
                IsInHaystack::Series(s) => {
                    let values = series_to_pyarrow_list(s)?;
                    Some(format!("({col}).isin({values})"))
                },
            }
        },
        PaIr::Between {
            column,
            left_op,
            lower,
            right_op,
            upper,
        } => {
            let column = render_pa_string(column)?;
            let left_symbol = binary_op_symbol(left_op)?;
            let right_symbol = binary_op_symbol(right_op)?;
            let lower = render_pa_string(lower)?;
            let upper = render_pa_string(upper)?;

            Some(format!(
                "(({column} {left_symbol} {lower}) & ({column} {right_symbol} {upper}))"
            ))
        },
    }
}

/// Whether the expression is known to be boolean without consulting the schema.
/// `^`, `&` and `|` are bitwise operations on integers, which PyArrow
/// expressions have no equivalent for.
fn returns_boolean(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    match expr_arena.get(node) {
        AExpr::BinaryExpr { left, right, op } => {
            op.is_comparison()
                || (op.is_bitwise()
                    && returns_boolean(*left, expr_arena)
                    && returns_boolean(*right, expr_arena))
        },
        AExpr::Literal(lv) => matches!(lv.get_datatype(), DataType::Boolean),
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            input,
            ..
        } => {
            // `Not` is also the bitwise negation.
            input
                .first()
                .is_some_and(|e| returns_boolean(e.node(), expr_arena))
        },
        AExpr::Function {
            function: IRFunctionExpr::Boolean(_),
            ..
        } => true,
        _ => false,
    }
}

/// Whether `left op right` is `Float64` arithmetic that PyArrow's checked
/// kernels evaluate exactly like Polars, so the minterm containing it can be
/// pushed down without an engine-side residual.
///
/// Only `+`, `-` and `*` with (transitively) `Float64` operands qualify:
///
/// * Integer arithmetic is excluded: Polars wraps on overflow where PyArrow's
///   `*_checked` kernels raise.
/// * `TrueDivide` is excluded: Polars yields `inf`/`NaN` on division by zero
///   where PyArrow raises, and matching Polars' float-division output dtype
///   would need a dividend cast that loses precision (`Float32`) or errors on
///   large integers.
/// * Anything but plain numeric operands (temporal, string, boolean, decimal,
///   casts, ...) is excluded.
fn is_float64_arithmetic(
    left: Node,
    right: Node,
    op: Operator,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> bool {
    if !matches!(op, Operator::Plus | Operator::Minus | Operator::Multiply) {
        return false;
    }

    fn operand_dtype(node: Node, expr_arena: &Arena<AExpr>, schema: &Schema) -> Option<DataType> {
        match expr_arena.get(node) {
            AExpr::Column(name) => schema.get(name).cloned(),
            AExpr::Literal(lv) => Some(lv.get_datatype()),
            AExpr::BinaryExpr { left, right, op }
                if is_float64_arithmetic(*left, *right, *op, expr_arena, schema) =>
            {
                Some(DataType::Float64)
            },
            _ => None,
        }
    }

    let (Some(left_dtype), Some(right_dtype)) = (
        operand_dtype(left, expr_arena, schema),
        operand_dtype(right, expr_arena, schema),
    ) else {
        return false;
    };

    // Both sides must be plain floats or integers, and at least one side must
    // already be `Float64` so the result is `Float64` rather than an integer
    // (wrapping) or `Float32` (different rounding) type.
    (left_dtype.is_float() || left_dtype.is_integer())
        && (right_dtype.is_float() || right_dtype.is_integer())
        && (matches!(left_dtype, DataType::Float64) || matches!(right_dtype, DataType::Float64))
}

/// `None` if `op` is arithmetic that PyArrow may evaluate differently than
/// Polars, `Some(())` otherwise (including for non-arithmetic operators).
/// Shared by both lowering paths so they cannot diverge on what is safe to
/// push (see [`is_float64_arithmetic`]).
fn reject_inexact_arithmetic(
    left: Node,
    right: Node,
    op: Operator,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<()> {
    if op.is_arithmetic() && !is_float64_arithmetic(left, right, op, expr_arena, schema) {
        None
    } else {
        Some(())
    }
}

/// The Python operator reproducing `op` on a PyArrow expression, or `None` when
/// PyArrow has no equivalent. Mirrors [`binary_op_method`].
///
/// The arithmetic operators map onto PyArrow's *checked* kernels; this is
/// exact only for the `Float64` cases admitted by [`is_float64_arithmetic`].
fn binary_op_symbol(op: &Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "==",
        Operator::NotEq => "!=",
        Operator::Lt => "<",
        Operator::LtEq => "<=",
        Operator::Gt => ">",
        Operator::GtEq => ">=",
        Operator::And | Operator::LogicalAnd => "&",
        Operator::Or | Operator::LogicalOr => "|",
        Operator::Plus => "+",
        Operator::Minus => "-",
        Operator::Multiply => "*",
        _ => return None,
    })
}

fn binary_op_method(op: &Operator) -> Option<&'static str> {
    Some(match op {
        Operator::Eq => "__eq__",
        Operator::NotEq => "__ne__",
        Operator::Lt => "__lt__",
        Operator::LtEq => "__le__",
        Operator::Gt => "__gt__",
        Operator::GtEq => "__ge__",
        Operator::And | Operator::LogicalAnd => "__and__",
        Operator::Or | Operator::LogicalOr => "__or__",
        Operator::Plus => "__add__",
        Operator::Minus => "__sub__",
        Operator::Multiply => "__mul__",
        _ => return None,
    })
}

// The main engine of converting AnyValue to a python object
fn anyvalue_to_py<'py>(py: Python<'py>, av: AnyValue<'_>) -> Option<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;

    let dtype = av.dtype();
    match av.as_borrowed() {
        AnyValue::Null => Some(py.None().into_bound(py)),
        AnyValue::Boolean(v) => v.into_bound_py_any(py).ok(),
        AnyValue::String(s) => s.into_pyobject(py).ok().map(|b| b.into_any()),
        #[cfg(feature = "dtype-date")]
        AnyValue::Date(days) => {
            use chrono::Datelike;
            let date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?
                .checked_add_signed(chrono::Duration::days(days as i64))?;
            PyDate::new(py, date.year(), date.month() as u8, date.day() as u8)
                .ok()
                .map(|b| b.into_any())
        },
        #[cfg(feature = "dtype-datetime")]
        AnyValue::Datetime(value, time_unit, time_zone) => {
            use chrono::{Datelike, Timelike};
            let micros: i64 = match time_unit {
                TimeUnit::Nanoseconds => (value % 1000 == 0).then_some(value / 1000)?,
                TimeUnit::Microseconds => value,
                TimeUnit::Milliseconds => value * 1000,
            };
            let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_micros(micros)?.naive_utc();
            let tzinfo: Option<Bound<'py, PyTzInfo>> = if let Some(tz) = time_zone {
                let zi = py.import("zoneinfo").ok()?;
                let obj = zi.getattr("ZoneInfo").ok()?.call1((tz.to_string(),)).ok()?;
                Some(obj.cast_into().ok()?)
            } else {
                None
            };
            PyDateTime::new(
                py,
                dt.year(),
                dt.month() as u8,
                dt.day() as u8,
                dt.hour() as u8,
                dt.minute() as u8,
                dt.second() as u8,
                dt.nanosecond() / 1000,
                tzinfo.as_ref(),
            )
            .ok()
            .map(|b| b.into_any())
        },
        // TODO: Worth supporting?
        AnyValue::Binary(_) | AnyValue::List(_) => None,
        #[cfg(feature = "dtype-array")]
        AnyValue::Array(_, _) => None,
        #[cfg(feature = "dtype-struct")]
        AnyValue::Struct(_, _, _) => None,
        av => {
            if dtype.is_float() {
                // TODO: Opportunity to downcast
                let v = av.extract::<f64>()?;
                v.into_bound_py_any(py).ok()
            } else if dtype.is_integer() {
                let v = av.extract::<i64>()?;
                v.into_bound_py_any(py).ok()
            } else {
                None
            }
        },
    }
}

fn series_to_py_list<'py>(py: Python<'py>, s: &Series) -> Option<Bound<'py, PyList>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(s.len());
    for av in s.iter() {
        items.push(anyvalue_to_py(py, av)?);
    }
    PyList::new(py, &items).ok()
}

// Convert an AExpr predicate to a pyarrow expression using python.
//
// `schema` is the scan output schema, used to resolve column dtypes so that
// only arithmetic provably equivalent to Polars' is lowered (see
// [`is_float64_arithmetic`]).
pub fn aexpr_to_pyarrow<'py>(
    py: Python<'py>,
    pc: &Bound<'py, PyAny>,
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> Option<Bound<'py, PyAny>> {
    render_pa_object(py, pc, &to_pa_ir(predicate, expr_arena, schema)?)
}

fn render_pa_object<'py>(
    py: Python<'py>,
    pc: &Bound<'py, PyAny>,
    ir: &PaIr,
) -> Option<Bound<'py, PyAny>> {
    match ir {
        PaIr::Field(name) => pc.call_method1("field", (name,)).ok(),
        PaIr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let val = anyvalue_to_py(py, av)?;
            pc.call_method1("scalar", (val,)).ok()
        },
        PaIr::BinOp { left, op, right } => {
            let method = binary_op_method(op)?;
            let l = render_pa_object(py, pc, left)?;
            let r = render_pa_object(py, pc, right)?;
            l.call_method1(method, (r,)).ok()
        },
        PaIr::Xor(l, r) => {
            let l = render_pa_object(py, pc, l)?;
            let r = render_pa_object(py, pc, r)?;
            let any = l.call_method1("__or__", (&r,)).ok()?;
            let both = l
                .call_method1("__and__", (&r,))
                .ok()?
                .call_method0("__invert__")
                .ok()?;

            any.call_method1("__and__", (both,)).ok()
        },
        // `scan_pyarrow_dataset` has never needed a validity-comparison or
        // starts_with translation; preserve that rather than newly enabling
        // it here, untested.
        PaIr::ValidityCompare { .. } | PaIr::StartsWith(..) => None,
        PaIr::Not(inner) => render_pa_object(py, pc, inner)?
            .call_method0("__invert__")
            .ok(),
        PaIr::IsNull(inner) => render_pa_object(py, pc, inner)?
            .call_method0("is_null")
            .ok(),
        PaIr::IsNotNull(inner) => render_pa_object(py, pc, inner)?
            .call_method0("is_null")
            .ok()?
            .call_method0("__invert__")
            .ok(),
        PaIr::IsNan(inner, valid) => {
            if !valid {
                return None;
            }
            render_pa_object(py, pc, inner)?.call_method0("is_nan").ok()
        },
        PaIr::IsNotNan(inner, valid) => {
            if !valid {
                return None;
            }
            render_pa_object(py, pc, inner)?
                .call_method0("is_nan")
                .ok()?
                .call_method0("__invert__")
                .ok()
        },
        #[cfg(feature = "is_in")]
        PaIr::IsIn(col, haystack) => {
            let col = render_pa_object(py, pc, col)?;
            let values_list = match haystack {
                IsInHaystack::Empty => return pc.call_method1("scalar", (false,)).ok(),
                IsInHaystack::Series(s) => series_to_py_list(py, s)?,
            };

            col.call_method1("isin", (values_list,)).ok()
        },
        PaIr::Between {
            column,
            left_op,
            lower,
            right_op,
            upper,
        } => {
            let column = render_pa_object(py, pc, column)?;
            let left_method = binary_op_method(left_op)?;
            let right_method = binary_op_method(right_op)?;
            let lower = render_pa_object(py, pc, lower)?;
            let upper = render_pa_object(py, pc, upper)?;

            let lower_cmp = column.call_method1(left_method, (lower,)).ok()?;
            let upper_cmp = column.call_method1(right_method, (upper,)).ok()?;
            lower_cmp.call_method1("__and__", (upper_cmp,)).ok()
        },
    }
}
