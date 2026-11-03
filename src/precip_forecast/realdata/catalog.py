"""Read-only, bounded NetCDF variable catalog for the V2 sample release."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .pseudotime import extract_time_token


REGIONAL_NWP_VARIABLES = (
    "CREF",
    "DCAPE",
    "MSL",
    "MXCREF",
    "MXU10",
    "MXV10",
    "PRS",
    "RH2M",
    "SF",
    "T2M",
    "TP",
    "TS",
    "U10",
    "UH",
    "V10",
    "VWS",
    "VWSDIR",
)

GLOBAL_NWP_VARIABLES = (
    "CPE",
    "D2M",
    "GPH",
    "HT0",
    "LCC",
    "LPE",
    "MN2T3",
    "MSL",
    "MX2T3",
    "POV",
    "PRS",
    "PTYPE",
    "RDI",
    "RHU",
    "SHU",
    "SST",
    "T2M",
    "TCC",
    "TCW",
    "TCWV",
    "TEM",
    "TP",
    "U10",
    "U100",
    "V10",
    "V100",
    "VVP",
    "WIU",
    "WIV",
)

OBSERVATION_VARIABLES: Mapping[str, tuple[str, ...]] = {
    "precipitation": ("PRE_HOR",),
    "radar": ("REF",),
    "convective": ("cimask",),
    "satellite": ("IR1", "IR2", "IR3", "IR4", "VIS"),
}

MODALITY_DIRECTORIES: Mapping[str, str] = {
    "precipitation": "SURF_PRE",
    "radar": "RADA_REF",
    "convective": "NMIC_CIDS",
    "satellite": "STAE_FY2G",
    "regional_nwp": "NWP_REG",
    "global_nwp": "NWP_GLB",
}

CATALOG_COLUMNS = (
    "status",
    "modality",
    "catalog_variable",
    "data_variable",
    "split",
    "process",
    "representative_path",
    "file_pattern",
    "file_size_bytes",
    "file_mtime_utc",
    "dimensions_json",
    "shape_json",
    "pressure_levels_json",
    "dtype",
    "units",
    "long_name",
    "standard_name",
    "fill_value_json",
    "missing_value_json",
    "scale_factor_json",
    "add_offset_json",
    "chunks_json",
    "compression_json",
    "forecast_reference_time",
    "valid_time_first",
    "valid_time_last",
    "valid_time_interval_seconds",
    "forecast_range",
    "sample_strategy",
    "sample_points",
    "sample_finite_ratio",
    "sample_masked_count",
    "sample_min",
    "sample_max",
    "sample_mean",
    "sample_q01",
    "sample_q50",
    "sample_q99",
    "open_seconds",
    "sample_read_seconds",
    "semantics_status",
    "error",
)


_NWP_FILENAME_RE = re.compile(
    r"^NWP_(?P<family>REG|GLB)_M\d+_(?P<variable>.+?)_"
    r"(?P<grid>0P\d+)_(?P<vertical>SURF|PLEV)_(?P<token>\d{10})\.NC$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CatalogTarget:
    modality: str
    catalog_variable: str
    path: Path | None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _json_cell(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _masked_float_values(sample: Any) -> tuple[Any, Any]:
    """Return a boolean mask and flattened float values with missing data as NaN."""

    import numpy as np

    masked = np.ma.asarray(sample).astype(np.float64)
    mask = np.ma.getmaskarray(masked)
    values = np.asarray(masked.filled(np.nan), dtype=np.float64).reshape(-1)
    return mask, values


def atomic_write_text(path: Path, text: str) -> None:
    """Write a small text artifact with fsync + same-directory replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def ensure_output_outside_data_root(data_root: Path, output: Path) -> None:
    data = Path(data_root).resolve()
    target = Path(output).resolve()
    try:
        target.relative_to(data)
    except ValueError:
        return
    raise ValueError(f"refusing to write an audit artifact inside read-only data root: {target}")


def parse_nwp_filename(path_or_name: str | Path) -> dict[str, str]:
    name = Path(path_or_name).name
    match = _NWP_FILENAME_RE.fullmatch(name)
    if not match:
        raise ValueError(f"not a recognized NWP sample filename: {name}")
    result = {key: value.upper() for key, value in match.groupdict().items()}
    # Validate the pseudo year/calendar token as part of parsing.
    result["token"] = extract_time_token(name)
    return result


def _sample_rank(path: Path, data_root: Path) -> tuple[int, str]:
    relative = path.relative_to(data_root).as_posix()
    if relative.startswith("TRAIN_DATA/P0001/"):
        return (0, relative)
    if relative.startswith("TRAIN_DATA/P0002/"):
        return (1, relative)
    if relative.startswith("VAL_DATA/V0001/"):
        return (2, relative)
    if relative.startswith("VAL_DATA/V0002/"):
        return (3, relative)
    return (4, relative)


def _iter_modality_files(data_root: Path, directory_name: str) -> Iterable[Path]:
    """Depth-bounded traversal of the documented split/process/modality layout."""

    for split_name in ("TRAIN_DATA", "VAL_DATA"):
        split = data_root / split_name
        if not split.is_dir():
            continue
        for process in sorted(item for item in split.iterdir() if item.is_dir()):
            modality = process / directory_name
            if not modality.is_dir():
                continue
            for path in sorted(modality.iterdir()):
                if path.is_file() and path.suffix.lower() in {".nc", ".nc4"}:
                    yield path


def discover_catalog_targets(data_root: Path) -> list[CatalogTarget]:
    """Choose one deterministic representative for every required variable."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    targets: list[CatalogTarget] = []
    for modality, variables in OBSERVATION_VARIABLES.items():
        files = sorted(
            _iter_modality_files(root, MODALITY_DIRECTORIES[modality]),
            key=lambda path: _sample_rank(path, root),
        )
        representative = files[0] if files else None
        targets.extend(CatalogTarget(modality, variable, representative) for variable in variables)

    for modality, expected_variables in (
        ("regional_nwp", REGIONAL_NWP_VARIABLES),
        ("global_nwp", GLOBAL_NWP_VARIABLES),
    ):
        representatives: dict[str, Path] = {}
        for path in _iter_modality_files(root, MODALITY_DIRECTORIES[modality]):
            try:
                parsed = parse_nwp_filename(path)
            except ValueError:
                continue
            variable = parsed["variable"]
            previous = representatives.get(variable)
            if previous is None or _sample_rank(path, root) < _sample_rank(previous, root):
                representatives[variable] = path
        ordered = list(expected_variables)
        ordered.extend(sorted(set(representatives) - set(expected_variables)))
        targets.extend(
            CatalogTarget(modality, variable, representatives.get(variable)) for variable in ordered
        )
    return targets


def bounded_sample_slices(shape: Sequence[int], max_points: int) -> tuple[slice, ...]:
    """Return one bounded central slab without allocating a full variable.

    At most three positions are read from a leading time dimension.  Other
    non-spatial axes (for example pressure/height) are retained when the
    budget allows.  The final two axes are represented by a central rectangle.
    """

    if max_points < 1:
        raise ValueError("max_points must be positive")
    dimensions = tuple(int(length) for length in shape)
    if any(length < 0 for length in dimensions):
        raise ValueError("shape lengths must be non-negative")
    if not dimensions:
        return ()
    counts = list(dimensions)
    if len(counts) >= 3:
        counts[0] = min(counts[0], 3)
    if len(counts) >= 2:
        # Preserve the leading time/level representation when possible and pay
        # the bounded-I/O budget primarily by shrinking the spatial slab.
        while math.prod(counts[:-2]) > max_points:
            reducible = [index for index in range(len(counts) - 2) if counts[index] > 1]
            if not reducible:
                break
            index = max(reducible, key=lambda item: counts[item])
            counts[index] = max(1, counts[index] // 2)
        prefix = max(1, math.prod(counts[:-2]))
        side_budget = max(1, max_points // prefix)
        ratio = dimensions[-2] / max(1, dimensions[-1])
        height = min(dimensions[-2], max(1, int(math.sqrt(side_budget * ratio))))
        width = min(dimensions[-1], max(1, side_budget // height))
        while prefix * height * width > max_points and (height > 1 or width > 1):
            if width >= height and width > 1:
                width -= 1
            elif height > 1:
                height -= 1
        counts[-2], counts[-1] = height, width
    elif math.prod(counts) > max_points:
        counts[0] = min(counts[0], max_points)

    slices = []
    for length, count in zip(dimensions, counts):
        start = max(0, (length - count) // 2)
        slices.append(slice(start, start + count, 1))
    return tuple(slices)


def _find_data_variable(dataset: Any, requested: str) -> tuple[str | None, str | None]:
    if requested in dataset.variables:
        return requested, None
    case_matches = [name for name in dataset.variables if name.upper() == requested.upper()]
    if len(case_matches) == 1:
        return case_matches[0], None
    candidates = []
    for name, variable in dataset.variables.items():
        standard_name = str(getattr(variable, "standard_name", "")).lower()
        axis = str(getattr(variable, "axis", "")).upper()
        coordinate_like = (
            name in dataset.dimensions
            or standard_name in {"time", "latitude", "longitude"}
            or axis in {"T", "X", "Y", "Z"}
        )
        if not coordinate_like and len(variable.dimensions) >= 2:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0], f"requested variable {requested!r} absent; unique data variable used"
    return None, f"requested variable {requested!r} absent; candidates={candidates!r}"


def _format_netcdf_time(value: Any) -> str:
    if all(hasattr(value, name) for name in ("year", "month", "day", "hour")):
        return (
            f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}T"
            f"{int(value.hour):02d}:{int(getattr(value, 'minute', 0)):02d}:"
            f"{int(getattr(value, 'second', 0)):02d}"
        )
    return str(value)


def _time_summary(dataset: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid_time_first": "",
        "valid_time_last": "",
        "valid_time_interval_seconds": "",
    }
    candidate = None
    for name, variable in dataset.variables.items():
        if (
            name.lower() in {"time", "valid_time", "forecast_time"}
            or str(getattr(variable, "standard_name", "")).lower() == "time"
            or str(getattr(variable, "axis", "")).upper() == "T"
        ):
            candidate = variable
            break
    if candidate is None or candidate.size == 0:
        return result
    try:
        raw = candidate[:]
        flat = raw.compressed() if hasattr(raw, "compressed") else raw.reshape(-1)
        if len(flat) == 0:
            return result
        units = getattr(candidate, "units", None)
        calendar = getattr(candidate, "calendar", "standard")
        if units:
            from netCDF4 import num2date

            decoded = num2date(
                [flat[0], flat[-1]],
                units=units,
                calendar=calendar,
                only_use_cftime_datetimes=True,
            )
            result["valid_time_first"] = _format_netcdf_time(decoded[0])
            result["valid_time_last"] = _format_netcdf_time(decoded[-1])
            if len(flat) > 1:
                first_two = num2date(
                    [flat[0], flat[1]],
                    units=units,
                    calendar=calendar,
                    only_use_cftime_datetimes=True,
                )
                result["valid_time_interval_seconds"] = float(
                    (first_two[1] - first_two[0]).total_seconds()
                )
        else:
            result["valid_time_first"] = str(_json_safe(flat[0]))
            result["valid_time_last"] = str(_json_safe(flat[-1]))
    except Exception as exc:  # evidence is kept in the row without losing the catalog
        result["valid_time_first"] = f"unreadable:{type(exc).__name__}:{exc}"
    return result


def _pressure_levels(dataset: Any, variable: Any) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for dimension in variable.dimensions:
        lower = dimension.lower()
        if not any(marker in lower for marker in ("level", "lev", "plev", "isobar", "height")):
            continue
        coordinate = dataset.variables.get(dimension)
        if coordinate is None:
            levels[dimension] = {"count": len(dataset.dimensions[dimension]), "values": None}
            continue
        try:
            values = coordinate[:]
            values = values.compressed() if hasattr(values, "compressed") else values.reshape(-1)
            safe_values = [_json_safe(item) for item in values[:100]]
            levels[dimension] = {
                "count": int(coordinate.size),
                "units": str(getattr(coordinate, "units", "unknown_pending_doc")),
                "values": safe_values,
                "truncated": int(coordinate.size) > 100,
            }
        except Exception as exc:
            levels[dimension] = {"error": f"{type(exc).__name__}: {exc}"}
    return levels


def _path_context(path: Path, data_root: Path) -> tuple[str, str]:
    relative = path.relative_to(data_root)
    parts = relative.parts
    if len(parts) < 2:
        return "unknown", "unknown"
    return parts[0], parts[1]


def inspect_catalog_target(
    target: CatalogTarget,
    data_root: Path,
    *,
    max_sample_points: int = 20_000,
) -> dict[str, Any]:
    row = {column: "" for column in CATALOG_COLUMNS}
    row.update(
        {
            "status": "missing_file" if target.path is None else "pending",
            "modality": target.modality,
            "catalog_variable": target.catalog_variable,
            "semantics_status": "metadata_only_no_unit_guessing",
        }
    )
    if target.path is None:
        row["error"] = "no representative file found in documented bounded layout"
        return row

    path = target.path.resolve()
    stat = path.stat()
    split, process = _path_context(path, data_root)
    row.update(
        {
            "split": split,
            "process": process,
            "representative_path": str(path),
            "file_pattern": re.sub(r"\d{10}(?=\.NC$)", "<YYYYMMDDHH>", path.name),
            "file_size_bytes": stat.st_size,
            "file_mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
    )

    try:
        from netCDF4 import Dataset

        start_open = time.perf_counter()
        with Dataset(str(path), mode="r") as dataset:
            row["open_seconds"] = time.perf_counter() - start_open
            variable_name, warning = _find_data_variable(dataset, target.catalog_variable)
            if variable_name is None:
                row["status"] = "missing_variable"
                row["error"] = warning or "data variable not found"
                return row
            variable = dataset.variables[variable_name]
            row.update(
                {
                    "data_variable": variable_name,
                    "dimensions_json": _json_cell(list(variable.dimensions)),
                    "shape_json": _json_cell(list(variable.shape)),
                    "pressure_levels_json": _json_cell(_pressure_levels(dataset, variable)),
                    "dtype": str(variable.dtype),
                    "units": str(getattr(variable, "units", "unknown_pending_doc")),
                    "long_name": str(getattr(variable, "long_name", "unknown_pending_doc")),
                    "standard_name": str(
                        getattr(variable, "standard_name", "unknown_pending_doc")
                    ),
                    "fill_value_json": _json_cell(getattr(variable, "_FillValue", None)),
                    "missing_value_json": _json_cell(getattr(variable, "missing_value", None)),
                    "scale_factor_json": _json_cell(getattr(variable, "scale_factor", None)),
                    "add_offset_json": _json_cell(getattr(variable, "add_offset", None)),
                    "chunks_json": _json_cell(variable.chunking()),
                    "forecast_reference_time": str(
                        getattr(dataset, "forecast_reference_time", "unknown_pending_doc")
                    ),
                    "forecast_range": str(
                        getattr(dataset, "forecast_range", "unknown_pending_doc")
                    ),
                }
            )
            try:
                row["compression_json"] = _json_cell(variable.filters())
            except Exception as exc:
                row["compression_json"] = _json_cell({"unavailable": str(exc)})
            row.update(_time_summary(dataset))

            slices = bounded_sample_slices(variable.shape, max_sample_points)
            row["sample_strategy"] = "bounded_central_slab"
            start_read = time.perf_counter()
            sample = variable[slices]
            row["sample_read_seconds"] = time.perf_counter() - start_read
            import numpy as np

            # Cast before filling: integer MaskedArray (for example radar REF
            # with int16/_FillValue=-999) cannot represent a NaN fill value.
            mask, values = _masked_float_values(sample)
            finite = np.isfinite(values)
            finite_values = values[finite]
            row["sample_points"] = int(values.size)
            row["sample_masked_count"] = int(mask.reshape(-1).sum())
            row["sample_finite_ratio"] = float(finite.mean()) if values.size else 0.0
            if finite_values.size:
                quantiles = np.quantile(finite_values, [0.01, 0.5, 0.99])
                row.update(
                    {
                        "sample_min": float(finite_values.min()),
                        "sample_max": float(finite_values.max()),
                        "sample_mean": float(finite_values.mean()),
                        "sample_q01": float(quantiles[0]),
                        "sample_q50": float(quantiles[1]),
                        "sample_q99": float(quantiles[2]),
                    }
                )
            row["status"] = "ok"
            if warning:
                row["error"] = warning
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def build_variable_catalog(
    data_root: Path,
    *,
    max_sample_points: int = 20_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(data_root).resolve()
    if not 100 <= max_sample_points <= 1_000_000:
        raise ValueError("max_sample_points must be between 100 and 1,000,000")
    targets = discover_catalog_targets(root)
    rows = [
        inspect_catalog_target(target, root, max_sample_points=max_sample_points)
        for target in targets
    ]
    status_counts = Counter(str(row["status"]) for row in rows)
    modality_status: dict[str, dict[str, int]] = {}
    for modality in MODALITY_DIRECTORIES:
        modality_status[modality] = dict(
            Counter(str(row["status"]) for row in rows if row["modality"] == modality)
        )
    missing_expected = {
        modality: [
            str(row["catalog_variable"])
            for row in rows
            if row["modality"] == modality and row["status"] != "ok"
        ]
        for modality in MODALITY_DIRECTORIES
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "traversal": "documented layout only; split/process/modality/file",
        "max_sample_points_per_variable": max_sample_points,
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "modality_status": modality_status,
        "missing_or_failed_variables": missing_expected,
        "required_nwp_counts": {
            "regional_nwp": len(REGIONAL_NWP_VARIABLES),
            "global_nwp": len(GLOBAL_NWP_VARIABLES),
        },
        "complete": all(row["status"] == "ok" for row in rows),
        "warning": (
            "Samples are bounded central slabs. Units and semantics are copied from source "
            "metadata; unknown fields remain unknown_pending_doc."
        ),
    }
    return rows, summary


def _catalog_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CATALOG_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CATALOG_COLUMNS})
    return buffer.getvalue()


def _catalog_markdown(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    lines = [
        "# V2 Sample Variable Catalog",
        "",
        f"- Data root: `{summary['data_root']}`",
        f"- Catalog rows: `{summary['row_count']}`",
        f"- Complete: `{str(summary['complete']).lower()}`",
        f"- Status counts: `{_json_cell(summary['status_counts'])}`",
        f"- Bounded sample points per variable: `{summary['max_sample_points_per_variable']}`",
        "- Semantics rule: source metadata only; no abbreviation-based unit guesses.",
        "",
        "| modality | variable | source variable | shape | units | finite ratio | status |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        finite = row.get("sample_finite_ratio", "")
        finite_text = f"{float(finite):.6f}" if finite != "" else ""
        lines.append(
            "| {modality} | {catalog_variable} | {data_variable} | `{shape}` | {units} | "
            "{finite} | {status} |".format(
                modality=row.get("modality", ""),
                catalog_variable=row.get("catalog_variable", ""),
                data_variable=row.get("data_variable", ""),
                shape=row.get("shape_json", ""),
                units=str(row.get("units", "")).replace("|", "\\|"),
                finite=finite_text,
                status=row.get("status", ""),
            )
        )
    failures = [row for row in rows if row.get("status") != "ok"]
    lines.extend(["", "## Missing or failed", ""])
    if failures:
        lines.extend(
            f"- `{row['modality']}/{row['catalog_variable']}`: {row.get('error', '')}"
            for row in failures
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "The finite ratios and quantiles describe bounded central slabs, not full-file "
            "quality. Missingness decisions require the dedicated full audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_variable_catalog(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    data_root = Path(str(summary["data_root"]))
    ensure_output_outside_data_root(data_root, directory)
    csv_path = directory / "variable_catalog.csv"
    md_path = directory / "variable_catalog.md"
    atomic_write_text(csv_path, _catalog_csv(rows))
    atomic_write_text(md_path, _catalog_markdown(rows, summary))
    return csv_path, md_path
