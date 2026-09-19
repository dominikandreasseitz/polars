from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Literal

from polars._utils.unstable import issue_unstable_warning
from polars._utils.wrap import wrap_ldf
from polars.io.cloud._utils import NoPickleOption
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    IcebergCatalogTableDescriptor,
    IcebergScanResolver,
    IcebergScanTableSerializer,
    IcebergTableWrap,
)

if TYPE_CHECKING:
    import pyiceberg.catalog
    import pyiceberg.expressions
    import pyiceberg.table

    import polars.io.iceberg
    from polars._typing import StorageOptionsDict
    from polars.lazyframe.frame import LazyFrame


def scan_iceberg(
    source: str | pyiceberg.table.Table,
    *,
    snapshot_id: int | None = None,
    from_snapshot_id_exclusive: int | None = None,
    to_snapshot_id_inclusive: int | None = None,
    storage_options: StorageOptionsDict | None = None,
    catalog: pyiceberg.catalog.Catalog
    | polars.io.iceberg.IcebergCatalogConfig
    | None = None,
    reader_override: Literal["native", "pyiceberg"] | None = None,
    use_metadata_statistics: bool = True,
    fast_deletion_count: bool | None = None,
    use_pyiceberg_filter: bool = True,
    row_filter: pyiceberg.expressions.BooleanExpression | None = None,
) -> LazyFrame:
    """
    Lazily read from an Apache Iceberg table.

    .. engine-support:: in-memory, streaming, distributed

    Parameters
    ----------
    source
        A PyIceberg table, or a 'namespace.table_name' identifier string,
        or an absolute path to the metadata.
    snapshot_id
        The snapshot ID to scan from.
    from_snapshot_id_exclusive
        The snapshot ID immediately before the first append snapshot to scan.
        Setting this or `to_snapshot_id_inclusive` enables an incremental append
        scan. If omitted, the scan starts from the oldest ancestor of the end
        snapshot.
    to_snapshot_id_inclusive
        The last snapshot ID to include in an incremental append scan. If omitted,
        the table's current snapshot is used.
    storage_options
        Extra options for the storage backends supported by `pyiceberg`.
        For cloud storages, this may include configurations for authentication etc.

        More info is available `here <https://py.iceberg.apache.org/configuration/>`__.
    catalog
        PyIceberg catalog to load the table from if the provided `target`
        was a table name.
    reader_override
        Overrides the reader used to read the data.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Note that this parameter should not be necessary outside of testing, as
        polars will by default automatically select the best reader.

        Available options:

        * native: Uses polars native reader. This allows for more optimizations to
          improve performance.
        * pyiceberg: Uses PyIceberg, which may support more features.
    use_metadata_statistics
        Whether to allow using statistics from Iceberg metadata files.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        When a filter is present, this allows using min/max statistics present
        in the Iceberg metadata files can be used to allow the reader to skip
        scanning of metadata from data files that are guaranteed to not match
        the filter.

        If a row-count is requested (i.e. `scan_iceberg().select(pl.len())`), this
        allows returning a count directly from Iceberg metadata. Note however that
        for datasets containing position delete files, `fast_deletion_count` must
        also be enabled for this to work.

    fast_deletion_count
        Allows returning a row count calculated directly from Iceberg metadata
        for datasets that contain position delete files. This will give incorrect
        results if position delete files contain duplicated entries.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
    use_pyiceberg_filter
        Convert and push the filter to PyIceberg where possible. This does not
        affect `row_filter`, which is always applied regardless of this
        setting.
    row_filter
        A PyIceberg `BooleanExpression` (see `pyiceberg.expressions
        <https://py.iceberg.apache.org/api/#row-filtering>`__) to apply
        directly to the table scan, bypassing the polars-to-PyIceberg
        predicate conversion. This is useful if you already have a PyIceberg
        filter expression on hand, or one that cannot be expressed through
        polars' predicate pushdown. It is combined with (ANDed to) any filter
        derived from `.filter()` calls on the returned `LazyFrame`.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Setting this forces the PyIceberg reader (as if
        `reader_override="pyiceberg"` were passed), since the native reader
        only uses PyIceberg filters for file-level pruning rather than
        row-level filtering. This means the native reader's other
        optimizations (metadata statistics pushdown, fast row counts, native
        deletion vector handling) are not available while `row_filter` is
        set. Combining `row_filter` with `reader_override="native"` raises
        `ValueError`.

    Returns
    -------
    LazyFrame

    Examples
    --------
    Creates a scan for an Iceberg table from local filesystem, or object store.

    >>> table_path = "file:/path/to/iceberg-table/metadata.json"
    >>> pl.scan_iceberg(table_path).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table from S3.
    See a list of supported storage options for S3 `here
    <https://py.iceberg.apache.org/configuration/#fileio>`__.

    >>> table_path = "s3://bucket/path/to/iceberg-table/metadata.json"
    >>> storage_options = {
    ...     "s3.region": "eu-central-1",
    ...     "s3.access-key-id": "THE_AWS_ACCESS_KEY_ID",
    ...     "s3.secret-access-key": "THE_AWS_SECRET_ACCESS_KEY",
    ... }
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table from Azure.
    Supported options for Azure are available `here
    <https://py.iceberg.apache.org/configuration/#azure-data-lake>`__.

    Following type of table paths are supported:

    * az://<container>/<path>/metadata.json
    * adl://<container>/<path>/metadata.json
    * abfs[s]://<container>/<path>/metadata.json

    >>> table_path = "az://container/path/to/iceberg-table/metadata.json"
    >>> storage_options = {
    ...     "adlfs.account-name": "AZURE_STORAGE_ACCOUNT_NAME",
    ...     "adlfs.account-key": "AZURE_STORAGE_ACCOUNT_KEY",
    ... }
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table from Google Cloud Storage.
    Supported options for GCS are available `here
    <https://py.iceberg.apache.org/configuration/#google-cloud-storage>`__.

    >>> table_path = "s3://bucket/path/to/iceberg-table/metadata.json"
    >>> storage_options = {
    ...     "gcs.project-id": "my-gcp-project",
    ...     "gcs.oauth.token": "ya29.dr.AfM...",
    ... }
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table with additional options.
    In the below example, `without_files` option is used which loads the table without
    file tracking information.

    >>> table_path = "/path/to/iceberg-table/metadata.json"
    >>> storage_options = {"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"}
    >>> pl.scan_iceberg(
    ...     table_path, storage_options=storage_options
    ... ).collect()  # doctest: +SKIP

    Creates a scan for an Iceberg table using a specific snapshot ID.

    >>> table_path = "/path/to/iceberg-table/metadata.json"
    >>> snapshot_id = 7051579356916758811
    >>> pl.scan_iceberg(table_path, snapshot_id=snapshot_id).collect()  # doctest: +SKIP

    Creates an incremental append scan between two snapshots.

    >>> pl.scan_iceberg(
    ...     table_path,
    ...     from_snapshot_id_exclusive=7051579356916758811,
    ...     to_snapshot_id_inclusive=8051579356916758811,
    ... ).collect()  # doctest: +SKIP
    """
    from polars._plr import PyLazyFrame

    if reader_override is not None:
        msg = "the `reader_override` parameter of `scan_iceberg()` is considered unstable."
        issue_unstable_warning(msg)

    if fast_deletion_count is not None:
        msg = "the `fast_deletion_count` parameter of `scan_iceberg()` is considered unstable."
        issue_unstable_warning(msg)
    else:
        fast_deletion_count = False

    if snapshot_id is not None and (
        from_snapshot_id_exclusive is not None or to_snapshot_id_inclusive is not None
    ):
        msg = (
            "cannot combine `snapshot_id` with `from_snapshot_id_exclusive` "
            "or `to_snapshot_id_inclusive`"
        )
        raise ValueError(msg)

    resolved_reader_override = reader_override

    if row_filter is not None:
        msg = "the `row_filter` parameter of `scan_iceberg()` is considered unstable."
        issue_unstable_warning(msg)

        if reader_override == "native":
            msg = (
                "`row_filter` is not supported together with `reader_override='native'`, "
                "since the native reader only uses PyIceberg filters for file-level "
                "pruning, not row-level filtering; either drop `reader_override` to let "
                "`row_filter` select the PyIceberg reader automatically, or express the "
                "condition as a `.filter()` call on the returned `LazyFrame` instead"
            )
            raise ValueError(msg)
        resolved_reader_override = "pyiceberg"

    table: pyiceberg.table.Table | None = None

    if importlib.util.find_spec("pyiceberg.table") is not None:
        import pyiceberg.table

        if isinstance(source, pyiceberg.table.Table):
            table = source

    table_descriptor_ = None

    if table is None:
        source = str(source)
        table_descriptor_ = (
            source  # Inferred as static metadata path
            if "/" in source or "\\" in source
            else IcebergCatalogTableDescriptor(
                table_identifier=source,
                catalog_config=IcebergCatalogConfig._from_api_parameter_or_environment_default(
                    catalog,
                    fn_name="scan_iceberg",
                ),
            )
        )

    dataset = IcebergScanResolver(
        table=IcebergTableWrap(
            table_=NoPickleOption(table),
            table_descriptor_=table_descriptor_,
            serializer=IcebergScanTableSerializer(),
            iceberg_storage_properties=storage_options,
        ),
        snapshot_id=snapshot_id,
        from_snapshot_id_exclusive=from_snapshot_id_exclusive,
        to_snapshot_id_inclusive=to_snapshot_id_inclusive,
        reader_override=resolved_reader_override,
        use_metadata_statistics=use_metadata_statistics,
        fast_deletion_count=fast_deletion_count,
        use_pyiceberg_filter=use_pyiceberg_filter,
        row_filter=row_filter,
    )

    return wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset))
