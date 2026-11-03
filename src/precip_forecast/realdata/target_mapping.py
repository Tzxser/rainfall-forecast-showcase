"""Coordinate-derived mappings between rectilinear grids.

``GridMap`` precomputes source fractional indices from latitude/longitude.  It
never derives a mapping from input/output shape ratios.  Every application also
returns the explicit target-grid outside mask so padding cannot be mistaken for
valid meteorological data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from precip_forecast.realdata.grids import GridSpec


def _fractional_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map coordinate values to fractional positions in the original axis."""

    positions = np.arange(source.size, dtype=np.float64)
    if source[-1] > source[0]:
        return np.interp(target, source, positions)
    return np.interp(target, source[::-1], positions[::-1])


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class GridMap:
    """A reusable latitude/longitude mapping from ``source`` to ``target``.

    ``bilinear`` is intended for continuous variables and ``nearest`` for
    categorical fields and masks.  Passing ``categorical=True`` enforces the
    latter rule.  Coordinates outside the source bounds are always exposed via
    :attr:`outside_mask`, independently of the optional output fill value.
    """

    source: GridSpec
    target: GridSpec
    method: str = "bilinear"
    categorical: bool = False
    tolerance: float = 1e-10
    _y_position: np.ndarray = field(init=False, repr=False)
    _x_position: np.ndarray = field(init=False, repr=False)
    _normalized_grid: np.ndarray = field(init=False, repr=False)
    _outside_mask: np.ndarray = field(init=False, repr=False)
    _mapping_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"bilinear", "nearest"}:
            raise ValueError("method must be 'bilinear' or 'nearest'")
        if self.categorical and self.method != "nearest":
            raise ValueError("categorical variables and masks require nearest mapping")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")

        y_position = _fractional_indices(self.source.latitude, self.target.latitude)
        x_position = _fractional_indices(self.source.longitude, self.target.longitude)
        lat_min, lat_max = self.source.latitude_bounds
        lon_min, lon_max = self.source.longitude_bounds
        outside_y = (self.target.latitude < lat_min - self.tolerance) | (
            self.target.latitude > lat_max + self.tolerance
        )
        outside_x = (self.target.longitude < lon_min - self.tolerance) | (
            self.target.longitude > lon_max + self.tolerance
        )
        outside = outside_y[:, None] | outside_x[None, :]

        y_normalized = 2.0 * y_position / (self.source.shape[0] - 1) - 1.0
        x_normalized = 2.0 * x_position / (self.source.shape[1] - 1) - 1.0
        x_mesh, y_mesh = np.meshgrid(x_normalized, y_normalized)
        normalized_grid = np.stack((x_mesh, y_mesh), axis=-1)

        hash_payload = {
            "schema": "precip-grid-map-v1",
            "source": self.source.coordinate_hash,
            "target": self.target.coordinate_hash,
            "method": self.method,
            "categorical": self.categorical,
            "tolerance": float(self.tolerance),
            "align_corners": True,
        }
        mapping_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        object.__setattr__(self, "_y_position", _readonly(y_position))
        object.__setattr__(self, "_x_position", _readonly(x_position))
        object.__setattr__(self, "_normalized_grid", _readonly(normalized_grid))
        object.__setattr__(self, "_outside_mask", _readonly(outside.astype(bool)))
        object.__setattr__(self, "_mapping_hash", mapping_hash)

    @property
    def outside_mask(self) -> np.ndarray:
        return self._outside_mask

    @property
    def normalized_grid(self) -> np.ndarray:
        """PyTorch ``grid_sample`` grid in ``(latitude, longitude, xy)`` order."""

        return self._normalized_grid

    @property
    def mapping_hash(self) -> str:
        return self._mapping_hash

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": "precip-grid-map-v1",
            "source_grid_hash": self.source.coordinate_hash,
            "target_grid_hash": self.target.coordinate_hash,
            "source_shape": list(self.source.shape),
            "target_shape": list(self.target.shape),
            "method": self.method,
            "categorical": self.categorical,
            "outside_count": int(np.count_nonzero(self.outside_mask)),
            "outside_fraction": float(np.mean(self.outside_mask)),
            "align_corners": True,
            "mapping_hash": self.mapping_hash,
        }

    def apply_numpy(
        self,
        values: np.ndarray,
        *,
        outside_fill: float | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the mapping to the final two dimensions of a NumPy array.

        When ``outside_fill`` is ``None``, boundary values are returned at
        outside points, but those points remain explicitly marked in the second
        return value.  Supplying a fill value replaces them after interpolation.
        """

        array = np.asarray(values)
        if array.ndim < 2 or tuple(array.shape[-2:]) != self.source.shape:
            raise ValueError(
                f"values final dimensions must be source shape {self.source.shape}, "
                f"got {array.shape}"
            )

        if self.method == "nearest":
            y_index = np.clip(
                np.floor(self._y_position + 0.5).astype(np.int64),
                0,
                self.source.shape[0] - 1,
            )
            x_index = np.clip(
                np.floor(self._x_position + 0.5).astype(np.int64),
                0,
                self.source.shape[1] - 1,
            )
            output = array[..., y_index[:, None], x_index[None, :]].copy()
        else:
            y0 = np.floor(self._y_position).astype(np.int64)
            x0 = np.floor(self._x_position).astype(np.int64)
            y0 = np.clip(y0, 0, self.source.shape[0] - 1)
            x0 = np.clip(x0, 0, self.source.shape[1] - 1)
            y1 = np.minimum(y0 + 1, self.source.shape[0] - 1)
            x1 = np.minimum(x0 + 1, self.source.shape[1] - 1)
            wy = (self._y_position - y0).reshape(
                (1,) * (array.ndim - 2) + (self.target.shape[0], 1)
            )
            wx = (self._x_position - x0).reshape(
                (1,) * (array.ndim - 2) + (1, self.target.shape[1])
            )
            top_left = array[..., y0[:, None], x0[None, :]]
            top_right = array[..., y0[:, None], x1[None, :]]
            bottom_left = array[..., y1[:, None], x0[None, :]]
            bottom_right = array[..., y1[:, None], x1[None, :]]
            output = (
                top_left * (1.0 - wy) * (1.0 - wx)
                + top_right * (1.0 - wy) * wx
                + bottom_left * wy * (1.0 - wx)
                + bottom_right * wy * wx
            )

        if outside_fill is not None and np.any(self.outside_mask):
            result_dtype = np.result_type(output.dtype, np.asarray(outside_fill).dtype)
            if isinstance(outside_fill, float) and np.isnan(outside_fill):
                result_dtype = np.result_type(result_dtype, np.float32)
            output = output.astype(result_dtype, copy=False)
            output[..., self.outside_mask] = outside_fill
        return output, self.outside_mask

    def apply_torch(self, values, *, outside_fill: float | int | None = None):
        """Apply with ``torch.nn.functional.grid_sample``.

        The last two dimensions are spatial; all preceding dimensions are
        treated as independent fields.  Float64 is preserved; every other
        input dtype (including float16, bfloat16, and integers) is promoted to
        float32 so the values and sampling grid always have one supported
        computation dtype.
        """

        import torch
        import torch.nn.functional as functional

        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim < 2 or tuple(values.shape[-2:]) != self.source.shape:
            raise ValueError(
                f"values final dimensions must be source shape {self.source.shape}, "
                f"got {tuple(values.shape)}"
            )
        leading_shape = tuple(values.shape[:-2])
        working = values
        if working.dtype != torch.float64 and working.dtype != torch.float32:
            working = working.to(dtype=torch.float32)
        batch = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1
        working = working.reshape(batch, 1, *self.source.shape)
        # GridMap deliberately exposes immutable NumPy evidence.  PyTorch warns
        # when ``as_tensor`` aliases such an array because tensor writes would
        # be undefined.  Make the ownership boundary explicit before transfer.
        writable_grid = np.array(self.normalized_grid, copy=True, order="C")
        grid = torch.as_tensor(
            writable_grid, dtype=working.dtype, device=working.device
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        output = functional.grid_sample(
            working,
            grid,
            mode=self.method,
            padding_mode="border",
            align_corners=True,
        ).reshape(*leading_shape, *self.target.shape)
        writable_outside = np.array(self.outside_mask, copy=True, order="C")
        outside = torch.as_tensor(
            writable_outside, dtype=torch.bool, device=working.device
        )
        if outside_fill is not None and bool(torch.any(outside)):
            expanded = outside.reshape((1,) * len(leading_shape) + self.target.shape)
            output = output.masked_fill(expanded, outside_fill)
        return output, outside
