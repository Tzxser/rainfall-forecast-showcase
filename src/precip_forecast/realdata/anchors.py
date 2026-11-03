"""Leakage-safe observation-only anchor construction for V2 sample events."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .catalog import atomic_write_text, ensure_output_outside_data_root
from .pseudotime import PseudoTime, add_hours, extract_time_token, iter_hour_tokens


WINDOW_OFFSETS: Mapping[str, tuple[int, ...]] = {
    "pre_input": tuple(range(-6, 0)),
    "radar_input": tuple(range(-3, 1)),
    "satellite_input": tuple(range(-6, 0)),
    "ci_input": tuple(range(-6, 1)),
    "target": (4, 5, 6),
}

WINDOW_MODALITIES: Mapping[str, str] = {
    "pre_input": "SURF_PRE",
    "radar_input": "RADA_REF",
    "satellite_input": "STAE_FY2G",
    "ci_input": "NMIC_CIDS",
    "target": "SURF_PRE",
}

ANCHOR_COLUMNS = (
    "sample_id",
    "process",
    "anchor_token",
    "anchor_time",
    "anchor_ordinal_hour",
    "eligible",
    "exclusion_reasons_json",
    "fold_a_role",
    "fold_b_role",
    "overfit_debug",
    "io_benchmark",
    "pre_input_paths_json",
    "radar_input_paths_json",
    "satellite_input_paths_json",
    "ci_input_paths_json",
    "target_paths_json",
    "target_leads_json",
    "nwp_manifest_status",
)


def _boolean_cell(value: Any) -> str:
    return "true" if bool(value) else "false"


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _discover_hour_files(directory: Path) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    by_token: dict[str, list[Path]] = defaultdict(list)
    anomalies: list[dict[str, str]] = []
    if not directory.is_dir():
        anomalies.append({"kind": "missing_modality_directory", "path": str(directory)})
        return {}, anomalies
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".nc", ".nc4"}:
            continue
        try:
            token = extract_time_token(path)
        except ValueError as exc:
            anomalies.append(
                {"kind": "unparseable_filename", "path": str(path), "detail": str(exc)}
            )
            continue
        by_token[token].append(path.resolve())
    for token, paths in sorted(by_token.items()):
        if len(paths) != 1:
            anomalies.append(
                {
                    "kind": "duplicate_hour_token",
                    "path": str(directory),
                    "detail": f"{token}: {[str(path) for path in paths]}",
                }
            )
    return dict(by_token), anomalies


def _resolve_window(
    inventory: Mapping[str, Mapping[str, Sequence[Path]]],
    field: str,
    anchor_token: str,
) -> tuple[list[str], list[str]]:
    modality = WINDOW_MODALITIES[field]
    files = inventory.get(modality, {})
    resolved: list[str] = []
    reasons: list[str] = []
    for offset in WINDOW_OFFSETS[field]:
        token = add_hours(anchor_token, offset)
        matches = list(files.get(token, ()))
        if len(matches) == 1:
            resolved.append(str(matches[0]))
        elif not matches:
            reasons.append(f"missing:{field}:{offset:+d}:{token}")
        else:
            reasons.append(f"duplicate:{field}:{offset:+d}:{token}:{len(matches)}")
    return resolved, reasons


def _fold_roles(process: str) -> tuple[str, str]:
    if process == "P0001":
        return "train", "validation"
    if process == "P0002":
        return "validation", "train"
    return "unassigned", "unassigned"


def _evenly_spaced_ids(rows: Sequence[Mapping[str, Any]], count: int) -> list[str]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return [str(row["sample_id"]) for row in rows]
    indexes = []
    for index in range(count):
        candidate = min(len(rows) - 1, ((2 * index + 1) * len(rows)) // (2 * count))
        if candidate not in indexes:
            indexes.append(candidate)
    # Integer rounding can only collide for unusual tiny inputs; fill deterministically.
    if len(indexes) < count:
        indexes.extend(item for item in range(len(rows)) if item not in indexes)
    return [str(rows[index]["sample_id"]) for index in sorted(indexes[:count])]


def select_fixed_subset(
    eligible_rows: Sequence[Mapping[str, Any]], count: int
) -> list[str]:
    """Select a deterministic, process-balanced, time-spread fixed subset."""

    if count < 0:
        raise ValueError("count cannot be negative")
    if count > len(eligible_rows):
        raise ValueError("requested subset is larger than eligible anchor population")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        grouped[str(row["process"])].append(row)
    processes = sorted(grouped)
    if not processes or count == 0:
        return []
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["anchor_token"]))
    allocation = {process: count // len(processes) for process in processes}
    for process in processes[: count % len(processes)]:
        allocation[process] += 1
    # Reallocate if a future sample release has a very small process.
    deficit = 0
    for process in processes:
        if allocation[process] > len(grouped[process]):
            deficit += allocation[process] - len(grouped[process])
            allocation[process] = len(grouped[process])
    while deficit:
        progressed = False
        for process in processes:
            if allocation[process] < len(grouped[process]):
                allocation[process] += 1
                deficit -= 1
                progressed = True
                if not deficit:
                    break
        if not progressed:
            raise ValueError("could not allocate fixed subset")
    selected = []
    for process in processes:
        selected.extend(_evenly_spaced_ids(grouped[process], allocation[process]))
    return selected


def build_anchor_index(
    data_root: Path,
    *,
    processes: Sequence[str] = ("P0001", "P0002"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all candidates and evidence for the strict observation windows.

    The returned row list contains both eligible and rejected hourly candidates.
    Writers default to placing only eligible rows in ``anchor_index.csv`` while
    retaining every rejected row and reason in the metadata JSON.
    """

    root = Path(data_root).resolve()
    training_root = root / "TRAIN_DATA"
    if not training_root.is_dir():
        raise ValueError(f"TRAIN_DATA is not a directory: {training_root}")
    all_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, str]] = []
    process_metadata: dict[str, Any] = {}

    for process in processes:
        process_root = training_root / process
        if not process_root.is_dir():
            process_metadata[process] = {
                "candidate_count": 0,
                "eligible_count": 0,
                "rejected_count": 0,
                "error": f"missing process directory: {process_root}",
            }
            continue
        inventory: dict[str, dict[str, list[Path]]] = {}
        for directory_name in sorted(set(WINDOW_MODALITIES.values())):
            files, found_anomalies = _discover_hour_files(process_root / directory_name)
            inventory[directory_name] = files
            anomalies.extend(
                {**item, "process": process, "modality": directory_name}
                for item in found_anomalies
            )
        candidate_tokens = sorted(
            set().union(*(set(files) for files in inventory.values())),
            key=lambda token: PseudoTime.parse(token),
        )
        fold_a_role, fold_b_role = _fold_roles(process)
        process_rows = []
        for token in candidate_tokens:
            windows: dict[str, list[str]] = {}
            reasons: list[str] = []
            for field in WINDOW_OFFSETS:
                paths, field_reasons = _resolve_window(inventory, field, token)
                windows[field] = paths
                reasons.extend(field_reasons)
            row: dict[str, Any] = {
                "sample_id": f"{process}_{token}",
                "process": process,
                "anchor_token": token,
                "anchor_time": PseudoTime.parse(token).isoformat(),
                "anchor_ordinal_hour": PseudoTime.parse(token).ordinal_hour(),
                "eligible": not reasons,
                "exclusion_reasons": reasons,
                "fold_a_role": fold_a_role,
                "fold_b_role": fold_b_role,
                "overfit_debug": False,
                "io_benchmark": False,
                "pre_input_paths": windows["pre_input"],
                "radar_input_paths": windows["radar_input"],
                "satellite_input_paths": windows["satellite_input"],
                "ci_input_paths": windows["ci_input"],
                "target_paths": windows["target"],
                "target_leads": list(WINDOW_OFFSETS["target"]),
                "nwp_manifest_status": "blocked_pending_time_gate",
            }
            process_rows.append(row)
        all_rows.extend(process_rows)
        expected_timeline_tokens: list[str] = []
        if candidate_tokens:
            expected_timeline_tokens = list(iter_hour_tokens(candidate_tokens[0], candidate_tokens[-1]))
        process_metadata[process] = {
            "candidate_count": len(process_rows),
            "eligible_count": sum(bool(row["eligible"]) for row in process_rows),
            "rejected_count": sum(not bool(row["eligible"]) for row in process_rows),
            "first_candidate_token": candidate_tokens[0] if candidate_tokens else None,
            "last_candidate_token": candidate_tokens[-1] if candidate_tokens else None,
            "expected_hour_count_from_endpoints": len(expected_timeline_tokens),
            "missing_hour_tokens": sorted(set(expected_timeline_tokens) - set(candidate_tokens)),
            "theoretical_complete_anchor_max_if_contiguous": max(
                0, len(expected_timeline_tokens) - 12
            ),
            "modality_file_counts": {
                modality: sum(len(paths) for paths in files.values())
                for modality, files in inventory.items()
            },
        }

    all_rows.sort(key=lambda row: (str(row["process"]), str(row["anchor_token"])))
    eligible = [row for row in all_rows if row["eligible"]]
    debug_ids = set(select_fixed_subset(eligible, min(4, len(eligible))))
    io_ids = set(select_fixed_subset(eligible, min(8, len(eligible))))
    for row in all_rows:
        row["overfit_debug"] = row["sample_id"] in debug_ids
        row["io_benchmark"] = row["sample_id"] in io_ids

    exact_reason_counts: Counter[str] = Counter()
    code_reason_counts: Counter[str] = Counter()
    rejected_candidates = []
    for row in all_rows:
        reasons = list(row["exclusion_reasons"])
        if reasons:
            rejected_candidates.append(
                {"sample_id": row["sample_id"], "anchor_token": row["anchor_token"], "reasons": reasons}
            )
        for reason in reasons:
            exact_reason_counts[reason] += 1
            code_reason_counts[":".join(reason.split(":")[:2])] += 1

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "training_root": str(training_root),
        "data_access": "read_only",
        "candidate_count": len(all_rows),
        "anchor_count": len(eligible),
        "actual_anchor_count": len(eligible),
        "eligible_anchor_count": len(eligible),
        "rejected_candidate_count": len(all_rows) - len(eligible),
        "targets_complete": all(len(row["target_paths"]) == 3 for row in eligible),
        "input_windows_complete": all(
            len(row["pre_input_paths"]) == 6
            and len(row["radar_input_paths"]) == 4
            and len(row["satellite_input_paths"]) == 6
            and len(row["ci_input_paths"]) == 7
            for row in eligible
        ),
        "window_offsets_hours": {key: list(value) for key, value in WINDOW_OFFSETS.items()},
        "window_modalities": dict(WINDOW_MODALITIES),
        "processes": process_metadata,
        "folds": {
            "fold_a": {"train": ["P0001"], "validation": ["P0002"]},
            "fold_b": {"train": ["P0002"], "validation": ["P0001"]},
        },
        "fixed_subsets": {
            "overfit_debug": sorted(debug_ids),
            "io_benchmark": sorted(io_ids),
        },
        "rejection_reason_code_counts": dict(sorted(code_reason_counts.items())),
        "rejection_reason_exact_counts": dict(sorted(exact_reason_counts.items())),
        "rejected_candidates": rejected_candidates,
        "inventory_anomalies": anomalies,
        "nwp_manifest_status": "blocked_pending_time_gate",
        "leakage_guards": [
            "PRE input ends at t-1",
            "radar and CI may include t but never t+1 or later",
            "satellite input ends at t-1",
            "targets are exactly t+4, t+5 and t+6",
            "NWP is absent until its independent availability-time gate passes",
        ],
    }
    return all_rows, metadata


def _rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ANCHOR_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "sample_id": row["sample_id"],
                "process": row["process"],
                "anchor_token": row["anchor_token"],
                "anchor_time": row["anchor_time"],
                "anchor_ordinal_hour": row["anchor_ordinal_hour"],
                "eligible": _boolean_cell(row["eligible"]),
                "exclusion_reasons_json": _json_cell(row["exclusion_reasons"]),
                "fold_a_role": row["fold_a_role"],
                "fold_b_role": row["fold_b_role"],
                "overfit_debug": _boolean_cell(row["overfit_debug"]),
                "io_benchmark": _boolean_cell(row["io_benchmark"]),
                "pre_input_paths_json": _json_cell(row["pre_input_paths"]),
                "radar_input_paths_json": _json_cell(row["radar_input_paths"]),
                "satellite_input_paths_json": _json_cell(row["satellite_input_paths"]),
                "ci_input_paths_json": _json_cell(row["ci_input_paths"]),
                "target_paths_json": _json_cell(row["target_paths"]),
                "target_leads_json": _json_cell(row["target_leads"]),
                "nwp_manifest_status": row["nwp_manifest_status"],
            }
        )
    return buffer.getvalue()


def write_anchor_outputs(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    output_dir: Path,
    *,
    include_rejected_in_csv: bool = False,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    ensure_output_outside_data_root(Path(str(metadata["data_root"])), directory)
    csv_rows = list(rows) if include_rejected_in_csv else [row for row in rows if row["eligible"]]
    enriched_metadata = dict(metadata)
    enriched_metadata["anchor_index_csv_includes_rejected"] = include_rejected_in_csv
    enriched_metadata["anchor_index_csv_row_count"] = len(csv_rows)
    csv_path = directory / "anchor_index.csv"
    metadata_path = directory / "anchor_index_metadata.json"
    atomic_write_text(csv_path, _rows_to_csv(csv_rows))
    atomic_write_text(
        metadata_path,
        json.dumps(enriched_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return csv_path, metadata_path


def read_anchor_index(path: Path, *, eligible_only: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in ("eligible", "overfit_debug", "io_benchmark"):
                row[field] = str(raw[field]).strip().lower() == "true"
            for field in (
                "exclusion_reasons",
                "pre_input_paths",
                "radar_input_paths",
                "satellite_input_paths",
                "ci_input_paths",
                "target_paths",
                "target_leads",
            ):
                row[field] = json.loads(raw[f"{field}_json"])
            row["anchor_ordinal_hour"] = int(raw["anchor_ordinal_hour"])
            if not eligible_only or row["eligible"]:
                rows.append(row)
    return rows
