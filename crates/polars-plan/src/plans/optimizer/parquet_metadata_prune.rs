//! Prune `FileScanIR::Parquet`'s pre-decoded `metadata_per_source` to
//! projected + predicate columns.
//!
//! Runs at the end of the optimizer pipeline, after projection and predicate
//! pushdown have resolved each scan's projection / predicate. Walks the IR
//! arena and replaces each entry via `FileMetadata::pruned`.
//!
//! Gated on `POLARS_PRUNE_PARQUET_METADATA=1` (default off; single-node
//! execution gets no benefit since metadata never crosses a wire).

#[cfg(feature = "parquet")]
use std::sync::Arc;

#[cfg(feature = "parquet")]
use polars_io::parquet::metadata::FileMetadataRef;
#[cfg(feature = "parquet")]
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};
#[cfg(feature = "parquet")]
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "parquet")]
use polars_utils::unitvec;

#[cfg(feature = "parquet")]
use crate::dsl::{ColumnMapping, FileScanIR};
#[cfg(feature = "parquet")]
use crate::plans::AExpr;
use crate::plans::IR;
#[cfg(feature = "parquet")]
use crate::utils::aexpr_to_leaf_names_iter;

#[cfg(feature = "parquet")]
pub(super) fn prune_parquet_metadata(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    // Gate on env var; default off (single-node has no benefit).
    if !polars_config::config().prune_parquet_metadata() {
        return;
    }

    let mut stack = unitvec![root];

    while let Some(node) = stack.pop() {
        ir_arena.get(node).copy_inputs(&mut stack);

        let IR::Scan {
            scan_type,
            unified_scan_args,
            predicate,
            ..
        } = ir_arena.get_mut(node)
        else {
            continue;
        };

        // Variant check first: skip the projection/predicate work entirely
        // for non-Parquet scans (CSV, IPC, NDJson, Python, Anonymous).
        let FileScanIR::Parquet {
            metadata_per_source,
            ..
        } = scan_type.as_mut()
        else {
            continue;
        };

        let Some(projection) = unified_scan_args.projection.clone() else {
            continue;
        };

        // Predicate cols are a subset of the projection by this point:
        // projection pushdown adds predicate-referenced cols to the scan
        // (else the pushed-down predicate would `ColumnNotFound` at execute).
        let predicate_cols: Vec<PlSmallStr> = predicate
            .as_ref()
            .map(|pred_ir| {
                aexpr_to_leaf_names_iter(pred_ir.node(), expr_arena)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // A renamed column's on-disk name differs from its output name, so pruning by
        // output name would drop it. Iceberg field IDs are stable across renames:
        // resolve the output names to field IDs once here, then re-resolve those IDs
        // to each file's own (possibly renamed) names inside `prune_one`.
        let field_ids = match &unified_scan_args.column_mapping {
            None => None,
            Some(ColumnMapping::Iceberg { schema, .. }) => {
                let name_to_id: PlHashMap<&str, i32> = schema
                    .values()
                    .filter_map(|c| Some((c.name.as_str(), i32::try_from(c.physical_id).ok()?)))
                    .collect();
                let resolve = |names: &[PlSmallStr]| -> Vec<i32> {
                    names
                        .iter()
                        .filter_map(|n| name_to_id.get(n.as_str()).copied())
                        .collect()
                };
                Some((resolve(&projection), resolve(&predicate_cols)))
            },
        };

        // Pruning failure (chunks-vs-leaves desync inside `from_compact`) is
        // recoverable; fall back to unpruned metadata so the query still runs.
        // The wire form just stays larger for this entry. Swap-only-if-smaller
        // (leaf count) avoids allocating a new inner Arc when nothing changed.
        let prune_one = |m: &FileMetadataRef| -> FileMetadataRef {
            let (keep, predicate_names) = match &field_ids {
                None => (projection.to_vec(), predicate_cols.clone()),
                Some((keep_ids, predicate_ids)) => {
                    // Files added without Parquet field IDs (e.g. Iceberg `add_files`)
                    // are matched by the reader via name mapping, not field ID; ID-based
                    // pruning isn't possible for them. Leave such files unpruned rather
                    // than resolve to an empty keep list and prune every column away.
                    let Some(keep) = names_for_field_ids(&m.schema_descr, keep_ids) else {
                        return m.clone();
                    };
                    let predicate_names =
                        names_for_field_ids(&m.schema_descr, predicate_ids).unwrap_or_default();
                    (keep, predicate_names)
                },
            };

            m.pruned(&keep, &predicate_names)
                .ok()
                .filter(|p| p.schema_descr.columns().len() < m.schema_descr.columns().len())
                .map(Arc::new)
                .unwrap_or_else(|| m.clone())
        };

        *metadata_per_source = metadata_per_source.map_footers(prune_one);
    }
}

/// Resolves Iceberg field IDs to this file's own top-level column names, which may
/// differ from the output names after a rename. Returns `None` if this file carries no
/// Parquet field IDs at all (e.g. added via Iceberg `add_files`, matched by the reader
/// through name mapping instead) - ID-based resolution isn't possible then, and the
/// caller must not treat that as "nothing to keep". IDs absent from an ID-bearing file
/// (e.g. a field added after it was written) are dropped individually, same as an
/// absent name.
#[cfg(feature = "parquet")]
fn names_for_field_ids(
    schema_descr: &polars_parquet::parquet::metadata::SchemaDescriptor,
    ids: &[i32],
) -> Option<Vec<PlSmallStr>> {
    let id_to_name: PlHashMap<i32, &PlSmallStr> = schema_descr
        .fields()
        .iter()
        .filter_map(|f| {
            let info = f.get_field_info();
            Some((info.id?, &info.name))
        })
        .collect();
    if id_to_name.is_empty() {
        return None;
    }
    Some(
        ids.iter()
            .filter_map(|id| id_to_name.get(id).map(|n| (*n).clone()))
            .collect(),
    )
}

#[cfg(all(test, feature = "parquet"))]
mod tests {
    use polars_parquet::parquet::metadata::SchemaDescriptor;
    use polars_parquet::parquet::schema::Repetition;
    use polars_parquet::parquet::schema::types::{ParquetType, PhysicalType};

    use super::names_for_field_ids;

    fn leaf(name: &str, id: Option<i32>) -> ParquetType {
        ParquetType::try_from_primitive(
            name.into(),
            PhysicalType::Int32,
            Repetition::Optional,
            None,
            None,
            id,
        )
        .unwrap()
    }

    /// `a` (id 1), `b` (no id), `c` (id 3).
    fn schema() -> SchemaDescriptor {
        SchemaDescriptor::new(
            "root".into(),
            vec![leaf("a", Some(1)), leaf("b", None), leaf("c", Some(3))],
        )
    }

    #[test]
    fn resolves_known_ids_to_this_files_names() {
        assert_eq!(names_for_field_ids(&schema(), &[3, 1]).unwrap(), ["c", "a"]);
    }

    #[test]
    fn drops_ids_this_file_has_no_field_for() {
        assert_eq!(names_for_field_ids(&schema(), &[1, 2]).unwrap(), ["a"]);
    }

    #[test]
    fn none_when_file_has_no_field_ids_at_all() {
        // E.g. a Parquet file added to an Iceberg table via `add_files`: matched by
        // name mapping, not field ID. Must not be confused with "no columns to keep".
        let no_ids = SchemaDescriptor::new("root".into(), vec![leaf("a", None), leaf("b", None)]);
        assert_eq!(names_for_field_ids(&no_ids, &[1, 2]), None);
    }
}

#[cfg(not(feature = "parquet"))]
pub(super) fn prune_parquet_metadata(
    _root: Node,
    _ir_arena: &mut Arena<IR>,
    _expr_arena: &Arena<crate::plans::AExpr>,
) {
    // No-op when parquet feature is disabled.
}
