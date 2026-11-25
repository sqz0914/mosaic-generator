"""Mosaic construction: match every grid cell to a tile and rebuild the image.

`MosaicBuilder` holds the settings and runs the pipeline:

    prepare image -> cell colors -> tile matching -> reconstruction

There are no Python loops over cells or tiles. Each step replaces a slow section of the
Lab 1 loop that profiling found (512x512 image, 32x32 grid):

    cell colors      <- one `mean` call per cell                     24% of the loop
    tile matching    <- inner loop over the tiles for every cell     32%
    reconstruction   <- per-cell `tile @ LUMA`, clip and paste       32%

Example:
    >>> from mosaic_generator import MosaicBuilder, TileManager
    >>> builder = MosaicBuilder(TileManager("Halftone Dots", tile_px=8), grid_size=16)
    >>> builder.create_mosaic(np.zeros((128, 128, 3), np.uint8)).shape
    (128, 128, 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .config import LUMA
from .image_processor import (
    GridImage,
    cell_statistics,
    prepare_image,
    quantize_colors,
    segmented_view,
)
from .metrics import Scores, score
from .tile_manager import TileManager, color_features, get_tile_manager
from .utils import MosaicInputError, StageTimer, check_choice, check_float, check_int, scratch


@dataclass
class MosaicResult:
    """Everything the web app and the report need from one mosaic.

    Attributes
    ----------
    original : np.ndarray
        The resized and grid-cropped input image (the reference for the metrics).
    segmented : np.ndarray
        The grid itself: every cell filled with its mean color, same size as `original`.
    mosaic : np.ndarray
        The mosaic rendered with one tile pixel per input pixel, same size as `original`
        (used for the metrics, so every grid size is scored at the same resolution).
    mosaic_full : np.ndarray
        The mosaic at the requested tile resolution (what the user sees).
    indices : np.ndarray
        (n_rows, n_cols) int32 index of the tile chosen for each cell.
    n_rows, n_cols : int
        Grid layout.
    cell_px : int
        Cell side in the input image, in pixels.
    tile_px : int
        Tile side in `mosaic_full`, in pixels (smaller than requested when the output
        would be larger than `config.MAX_OUTPUT_PX`).
    stages_ms : dict of str to float
        Processing time of each step, in milliseconds.
    """

    original: np.ndarray
    segmented: np.ndarray
    mosaic: np.ndarray
    mosaic_full: np.ndarray
    indices: np.ndarray
    n_rows: int
    n_cols: int
    cell_px: int
    tile_px: int
    stages_ms: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Total processing time of the mosaic.

        Returns
        -------
        float
            Sum of `stages_ms`, in milliseconds.
        """
        return float(sum(self.stages_ms.values()))


# --------------------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------------------


def match_tiles(cell_mean: np.ndarray, cell_std: np.ndarray | None, tile_manager: TileManager,
                space: str = config.DEFAULT_MATCH_SPACE,
                detail_weight: float = config.DEFAULT_DETAIL_WEIGHT) -> np.ndarray:
    """Pick the nearest tile for every cell, all cells at once.

    One broadcast compares every cell with every tile, (rows, cols, 1, F) against
    (1, 1, N, F), and `argmin` picks the closest. The squared distance is used instead of
    `np.linalg.norm`, because the square root does not change which tile is nearest.
    `argmin` returns the first minimum, the same tie-break as the Lab 1 loop. With
    `detail_weight > 0`, cells with a lot of internal contrast are pulled toward busy
    tiles and flat cells toward flat tiles.

    Parameters
    ----------
    cell_mean : np.ndarray
        (n_rows, n_cols, 3) float32 cell colors.
    cell_std : np.ndarray or None
        (n_rows, n_cols) brightness contrast of each cell; needed when `detail_weight > 0`.
    tile_manager : TileManager
        The tiles to choose from (their keys are cached).
    space : str, default "RGB"
        "RGB" or "CIE-Lab".
    detail_weight : float, default 0.0
        Weight of the texture term in [0, 1]; 0 matches on color only.

    Returns
    -------
    np.ndarray
        (n_rows, n_cols) int32 tile index for every cell.

    Example
    -------
    >>> tm = TileManager("Halftone Dots", 8)
    >>> match_tiles(np.zeros((2, 2, 3), np.float32), None, tm).tolist()
    [[0, 0], [0, 0]]
    """
    key_feat = tile_manager.match_keys(space)                          # cached (N, F)
    cell_feat = color_features(cell_mean, tile_manager.mode, space)    # (rows, cols, F)

    diff = cell_feat[:, :, None, :] - key_feat[None, None, :, :]
    dist = np.einsum("rcnf,rcnf->rcn", diff, diff)
    dist /= 255.0 ** 2 * cell_feat.shape[-1]          # roughly [0, 1], to mix with detail

    if detail_weight > 0 and cell_std is not None:
        cell_detail = np.clip(cell_std / 96.0, 0, 1)
        detail_diff = cell_detail[:, :, None] - tile_manager.tile_detail()[None, None, :]
        dist = (1.0 - detail_weight) * dist + detail_weight * detail_diff ** 2

    return dist.argmin(axis=-1).astype(np.int32)


def cell_offset(cell_mean: np.ndarray, color_mode: str, strength: float) -> np.ndarray | None:
    """The per-cell part of the recolor (the tile part comes from `TileManager.render_base`).

    Parameters
    ----------
    cell_mean : np.ndarray
        (n_rows, n_cols, 3) float32 cell colors.
    color_mode : str
        One of `config.COLOR_MODES`.
    strength : float
        Recolor strength in [0, 1].

    Returns
    -------
    np.ndarray or None
        (n_rows, n_cols, 3) float32 color to add to each cell's tile: `cell * s` for
        Blend, `(cell - cell_luma) * s` for Modulate, or None for Tile only and
        strength 0.

    Example
    -------
    >>> cell_offset(np.full((1, 1, 3), 100.0, np.float32), "Blend", 0.5).tolist()
    [[[50.0, 50.0, 50.0]]]
    """
    if color_mode == "Tile only" or strength <= 0:
        return None
    if color_mode == "Blend":
        return cell_mean * strength
    cell_luma = (cell_mean @ LUMA)[..., None]
    return (cell_mean - cell_luma) * strength


def split_offset(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a signed color offset into rounded positive and negative uint8 parts.

    uint8 cannot hold negative numbers, so `tile + offset` is applied as a saturating add
    of the positive part followed by a saturating subtract of the negative part. In each
    channel at most one part is non-zero, so this gives exactly
    `clip(tile + round(offset), 0, 255)`.

    Parameters
    ----------
    offset : np.ndarray
        (..., 3) float offsets in [-255, 255].

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (positive, negative) uint8 arrays with the shape of `offset`.

    Example
    -------
    >>> pos, neg = split_offset(np.array([[[10.4, -3.6, 0.0]]]))
    >>> pos.tolist(), neg.tolist()
    ([[[10, 0, 0]]], [[[0, 4, 0]]])
    """
    pos = np.rint(np.clip(offset, 0, 255)).astype(np.uint8)
    neg = np.rint(np.clip(-offset, 0, 255)).astype(np.uint8)
    return pos, neg


def gather_tiles(tiles: np.ndarray, indices: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Lay out the chosen tiles as one image, copying whole tile rows at once.

    Output pixel row y, cell column c shows row `y % T` of tile `indices[y // T, c]`, so
    a small table of row numbers lets one `np.take` copy every tile row straight to its
    final place. This replaces gathering a 5-D array and then copying it again to
    transpose it. `mode="clip"` matters for speed: with the default mode NumPy buffers
    `out`, which was about 20 times slower. Every index is valid, so clipping never
    changes the result.

    Parameters
    ----------
    tiles : np.ndarray
        (N, T, T, 3) tile images.
    indices : np.ndarray
        (n_rows, n_cols) chosen tile for every cell.
    out : np.ndarray, optional
        Contiguous (n_rows*T, n_cols*T, 3) array of `tiles.dtype` to fill.

    Returns
    -------
    np.ndarray
        `out`, or a new (n_rows*T, n_cols*T, 3) array.

    Example
    -------
    >>> tiles = np.arange(2 * 1 * 1 * 3, dtype=np.uint8).reshape(2, 1, 1, 3)
    >>> gather_tiles(tiles, np.array([[1, 0]]))[0].tolist()
    [[3, 4, 5], [0, 1, 2]]
    """
    n_rows, n_cols = indices.shape
    t = tiles.shape[1]
    tile_rows = tiles.reshape(-1, t * 3)                                   # (N*T, T*3) view
    row_in_tile = np.tile(np.arange(t, dtype=np.intp), n_rows)[:, None]    # (n_rows*T, 1)
    row_index = np.repeat(indices, t, axis=0).astype(np.intp) * t + row_in_tile

    if out is None:
        out = np.empty((n_rows * t, n_cols * t, 3), dtype=tiles.dtype)
    np.take(tile_rows, row_index, axis=0, out=out.reshape(n_rows * t, n_cols, t * 3), mode="clip")
    return out


def _per_pixel(cell_values: np.ndarray, shape: tuple[int, int, int], tag: str) -> np.ndarray:
    """Repeat each cell's value over its T x T pixels, into a reused work buffer.

    A nearest-neighbor resize by a whole factor is an exact repetition.

    Parameters
    ----------
    cell_values : np.ndarray
        (n_rows, n_cols, 3) uint8 values.
    shape : tuple of int
        Output shape (n_rows*T, n_cols*T, 3).
    tag : str
        Name of the work buffer (see `utils.scratch`).

    Returns
    -------
    np.ndarray
        (n_rows*T, n_cols*T, 3) uint8 work buffer; overwritten by the next call.
    """
    h, w = shape[:2]
    return cv2.resize(cell_values, (w, h), dst=scratch(shape, cell_values.dtype, tag),
                      interpolation=cv2.INTER_NEAREST)


def reconstruct(indices: np.ndarray, tile_manager: TileManager, tile_px: int,
                cell_mean: np.ndarray, color_mode: str = config.DEFAULT_COLOR_MODE,
                strength: float = config.DEFAULT_STRENGTH) -> np.ndarray:
    """Paste the chosen, recolored tile into every cell, as one image.

    The tile part of the recolor comes from the cache (`TileManager.render_base`). The
    tiles are gathered straight into the output, and each cell's color offset is repeated
    over its pixels and applied with OpenCV's saturating add and subtract, which also
    clip to [0, 255]. All full-size work stays in uint8.

    Color modes:
      "Tile only" -- the raw tile.
      "Blend"     -- tile * (1-s) + cell * s.
      "Modulate"  -- keeps the tile's brightness and takes the cell's color:
                     tile_luma + (cell - luma(cell)), mixed in with strength s.

    Parameters
    ----------
    indices : np.ndarray
        (n_rows, n_cols) tile index of every cell.
    tile_manager : TileManager
        The tiles.
    tile_px : int
        Tile side in the output, in pixels.
    cell_mean : np.ndarray
        (n_rows, n_cols, 3) float32 cell colors.
    color_mode : str, default "Modulate"
        One of `config.COLOR_MODES`.
    strength : float, default 0.85
        Recolor strength in [0, 1].

    Returns
    -------
    np.ndarray
        New (n_rows*tile_px, n_cols*tile_px, 3) uint8 mosaic.

    Example
    -------
    >>> tm = TileManager("Halftone Dots", 8)
    >>> reconstruct(np.zeros((2, 3), np.int32), tm, 8, np.zeros((2, 3, 3), np.float32)).shape
    (16, 24, 3)
    """
    base = tile_manager.render_base(tile_px, color_mode, strength)    # cached, uint8
    out = gather_tiles(base, indices)

    offset = cell_offset(cell_mean, color_mode, strength)
    if offset is not None:
        pos, neg = split_offset(offset)
        cv2.add(out, _per_pixel(pos, out.shape, "offset+"), dst=out)       # stops at 255
        cv2.subtract(out, _per_pixel(neg, out.shape, "offset-"), dst=out)  # stops at 0
    return out


def fit_tile_px(tile_px: int, n_rows: int, n_cols: int) -> int:
    """Shrink the tile size if the mosaic would be larger than `config.MAX_OUTPUT_PX`.

    Parameters
    ----------
    tile_px : int
        Requested tile side in pixels.
    n_rows, n_cols : int
        Grid layout.

    Returns
    -------
    int
        `tile_px`, or the largest size (at least 4) that keeps the mosaic within the cap.

    Example
    -------
    >>> fit_tile_px(32, 128, 128)
    16
    """
    longest = max(n_rows, n_cols)
    if longest * tile_px <= config.MAX_OUTPUT_PX:
        return tile_px
    return max(4, config.MAX_OUTPUT_PX // longest)


# --------------------------------------------------------------------------------------
# MosaicBuilder
# --------------------------------------------------------------------------------------


class MosaicBuilder:
    """Build photo mosaics from one tile set and one group of settings.

    Parameters
    ----------
    tile_manager : TileManager
        The tiles to build the mosaic from. Its `tile_px` is the tile size in the output.
    grid_size : int or tuple of (int, int), default config.DEFAULT_GRID_SIZE
        Number of cells across the image. A (rows, cols) tuple is also accepted, as in
        `grid_size=(32, 32)`; cells are always square, so the number of rows follows from
        the image's shape and only `cols` is used.
    color_mode : str, default "Modulate"
        One of `config.COLOR_MODES`.
    strength : float, default 0.85
        How strongly each cell's color tints its tile, in [0, 1].
    match_space : str, default "RGB"
        Color space for tile matching, "RGB" or "CIE-Lab".
    detail_weight : float, default 0.0
        Weight of texture matching in [0, 1]; 0 matches on color only.
    quantize_k : int, default 0
        Reduce the cell colors to this many with k-means first; 0 or 1 turns it off.
    long_edge : int, default config.DEFAULT_LONG_EDGE
        Working resolution: the input is resized so its longest side has this length.

    Raises
    ------
    MosaicInputError
        If any setting is of the wrong type or out of range.

    Example
    -------
    >>> builder = MosaicBuilder(TileManager("LEGO Studs", tile_px=8), grid_size=(32, 32))
    >>> img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    >>> mosaic = builder.create_mosaic(img)
    >>> mosaic.shape
    (256, 256, 3)
    >>> round(builder.compute_similarity(img, mosaic).ssim, 1) < 1
    True
    """

    def __init__(self, tile_manager: TileManager, grid_size: int | tuple[int, int] = config.DEFAULT_GRID_SIZE,
                 color_mode: str = config.DEFAULT_COLOR_MODE, strength: float = config.DEFAULT_STRENGTH,
                 match_space: str = config.DEFAULT_MATCH_SPACE,
                 detail_weight: float = config.DEFAULT_DETAIL_WEIGHT,
                 quantize_k: int = config.DEFAULT_QUANTIZE_K,
                 long_edge: int = config.DEFAULT_LONG_EDGE) -> None:
        """Validate and store the settings.

        Parameters
        ----------
        tile_manager : TileManager
            The tiles.
        grid_size : int or tuple of (int, int)
            Cells across, or (rows, cols).
        color_mode : str
            One of `config.COLOR_MODES`.
        strength : float
            Recolor strength in [0, 1].
        match_space : str
            "RGB" or "CIE-Lab".
        detail_weight : float
            Texture weight in [0, 1].
        quantize_k : int
            Palette size, 0 or 1 for off.
        long_edge : int
            Working resolution in pixels.

        Returns
        -------
        None
        """
        if not isinstance(tile_manager, TileManager):
            raise MosaicInputError(
                f"tile_manager must be a TileManager, got {type(tile_manager).__name__}")
        if isinstance(grid_size, (tuple, list)):
            if len(grid_size) != 2:
                raise MosaicInputError(f"grid_size must be an int or a (rows, cols) pair, got {grid_size!r}")
            grid_size = grid_size[1]

        self.tile_manager = tile_manager
        self.grid_size = check_int("grid_size", grid_size, config.MIN_GRID_SIZE, config.MAX_GRID_SIZE)
        self.color_mode = check_choice("color_mode", color_mode, config.COLOR_MODES)
        self.strength = check_float("strength", strength, 0.0, 1.0)
        self.match_space = check_choice("match_space", match_space, config.MATCH_SPACES)
        self.detail_weight = check_float("detail_weight", detail_weight, 0.0, 1.0)
        self.quantize_k = check_int("quantize_k", quantize_k, 0, config.MAX_QUANTIZE_K)
        self.long_edge = check_int("long_edge", long_edge, config.MIN_LONG_EDGE, config.MAX_LONG_EDGE)

    def __repr__(self) -> str:
        """Short text form for printing and debugging.

        Returns
        -------
        str
            E.g. "MosaicBuilder(tiles='Halftone Dots', grid_size=32, color_mode='Modulate')".
        """
        return (f"MosaicBuilder(tiles={self.tile_manager.name!r}, grid_size={self.grid_size}, "
                f"color_mode={self.color_mode!r})")

    def build(self, image) -> MosaicResult:
        """Build a mosaic and return it with every intermediate result and the timings.

        Parameters
        ----------
        image : str, Path, np.ndarray or PIL.Image.Image
            The input image, or a path to it.

        Returns
        -------
        MosaicResult
            The mosaic, the cropped input, the grid view, the tile choices and the
            processing time of each step.

        Raises
        ------
        MosaicInputError
            If the image cannot be loaded or does not fit the grid.

        Example
        -------
        >>> result = MosaicBuilder(TileManager(tile_px=8), grid_size=16).build(np.zeros((64, 64, 3), np.uint8))
        >>> result.n_rows, result.n_cols, sorted(result.stages_ms)
        (16, 16, ['Cell colors', 'Preprocess', 'Reconstruction', 'Tile matching'])
        """
        timer = StageTimer()
        tm = self.tile_manager

        with timer.stage("Preprocess"):
            grid: GridImage = prepare_image(image, self.grid_size, self.long_edge)
        with timer.stage("Cell colors"):
            cell_mean, cell_std = cell_statistics(*grid, with_std=self.detail_weight > 0)
            if self.quantize_k > 1:
                cell_mean = quantize_colors(cell_mean.astype(np.uint8), self.quantize_k).astype(np.float32)
        with timer.stage("Tile matching"):
            indices = match_tiles(cell_mean, cell_std, tm, self.match_space, self.detail_weight)
        with timer.stage("Reconstruction"):
            tile_px = fit_tile_px(tm.tile_px, grid.n_rows, grid.n_cols)
            mosaic_full = reconstruct(indices, tm, tile_px, cell_mean, self.color_mode, self.strength)

        # For the metrics, render the same tile choices at one tile pixel per input pixel,
        # so every grid size is scored at the same resolution with no resampling.
        if tile_px == grid.cell_px:
            mosaic = mosaic_full
        else:
            mosaic = reconstruct(indices, tm, grid.cell_px, cell_mean, self.color_mode, self.strength)

        h, w = grid.image.shape[:2]
        return MosaicResult(
            original=grid.image,
            segmented=segmented_view(cell_mean, h, w),
            mosaic=mosaic,
            mosaic_full=mosaic_full,
            indices=indices,
            n_rows=grid.n_rows,
            n_cols=grid.n_cols,
            cell_px=grid.cell_px,
            tile_px=tile_px,
            stages_ms=timer.stages_ms,
        )

    def create_mosaic(self, image) -> np.ndarray:
        """Build a mosaic and return only the mosaic image.

        Parameters
        ----------
        image : str, Path, np.ndarray or PIL.Image.Image
            The input image, or a path to it.

        Returns
        -------
        np.ndarray
            (n_rows*tile_px, n_cols*tile_px, 3) uint8 mosaic.

        Raises
        ------
        MosaicInputError
            If the image cannot be loaded or does not fit the grid.

        Example
        -------
        >>> MosaicBuilder(TileManager(tile_px=8), grid_size=16).create_mosaic(np.zeros((64, 64, 3), np.uint8)).shape
        (128, 128, 3)
        """
        return self.build(image).mosaic_full

    def compute_similarity(self, image, mosaic: np.ndarray) -> Scores:
        """Compare an input image with a mosaic made from it.

        The input is prepared exactly as `build` does (resized and cropped to the grid).
        If the mosaic has a different size -- because its tiles are larger or smaller
        than the cells -- it is resized to the input's size with area averaging first.

        Parameters
        ----------
        image : str, Path, np.ndarray or PIL.Image.Image
            The original input image, or a path to it.
        mosaic : np.ndarray
            (H, W, 3) mosaic, e.g. from `create_mosaic`.

        Returns
        -------
        Scores
            MSE (0 = identical) and SSIM (1 = identical).

        Raises
        ------
        MosaicInputError
            If the image cannot be loaded or the mosaic is not an (H, W, 3) image.

        Example
        -------
        >>> builder = MosaicBuilder(TileManager(tile_px=8), grid_size=16)
        >>> img = np.zeros((64, 64, 3), np.uint8)
        >>> builder.compute_similarity(img, builder.create_mosaic(img)).mse >= 0
        True
        """
        reference = prepare_image(image, self.grid_size, self.long_edge).image
        mosaic = np.asarray(mosaic)
        if mosaic.ndim != 3 or mosaic.shape[2] != 3:
            raise MosaicInputError(f"mosaic must be an (H, W, 3) image, got shape {mosaic.shape}")
        if mosaic.shape != reference.shape:
            h, w = reference.shape[:2]
            mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_AREA)
        return score(reference, mosaic)


# --------------------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------------------


def build_mosaic(image, grid_size: int = config.DEFAULT_GRID_SIZE,
                 tile_set_name: str = config.DEFAULT_TILE_SET, tile_px: int = config.DEFAULT_TILE_PX,
                 color_mode: str = config.DEFAULT_COLOR_MODE, strength: float = config.DEFAULT_STRENGTH,
                 match_space: str = config.DEFAULT_MATCH_SPACE,
                 detail_weight: float = config.DEFAULT_DETAIL_WEIGHT,
                 quantize_k: int = config.DEFAULT_QUANTIZE_K,
                 long_edge: int = config.DEFAULT_LONG_EDGE) -> MosaicResult:
    """Build a mosaic with a built-in tile set in one call.

    Same arguments as the Lab 1 `build_mosaic`, so the notebook and the web app can call
    it directly. The tile manager is shared between calls (`get_tile_manager`), so its
    cached tile features are reused.

    Parameters
    ----------
    image : str, Path, np.ndarray or PIL.Image.Image
        The input image, or a path to it.
    grid_size : int, default config.DEFAULT_GRID_SIZE
        Number of cells across.
    tile_set_name : str, default config.DEFAULT_TILE_SET
        Name of a built-in tile set.
    tile_px : int, default config.DEFAULT_TILE_PX
        Tile resolution in the output, in pixels.
    color_mode : str, default "Modulate"
        One of `config.COLOR_MODES`.
    strength : float, default 0.85
        Recolor strength in [0, 1].
    match_space : str, default "RGB"
        "RGB" or "CIE-Lab".
    detail_weight : float, default 0.0
        Texture weight in [0, 1].
    quantize_k : int, default 0
        Palette size; 0 or 1 for off.
    long_edge : int, default config.DEFAULT_LONG_EDGE
        Working resolution in pixels.

    Returns
    -------
    MosaicResult
        The mosaic, its intermediate results and the timings.

    Raises
    ------
    MosaicInputError
        If any argument is invalid or the image cannot be used.

    Example
    -------
    >>> build_mosaic(np.zeros((512, 512, 3), np.uint8), grid_size=32, tile_px=16, long_edge=512).mosaic.shape
    (512, 512, 3)
    """
    tile_manager = get_tile_manager(check_choice("tile_set_name", tile_set_name,
                                                 TileManager.available_tile_sets()),
                                    check_int("tile_px", tile_px, config.MIN_TILE_PX, config.MAX_TILE_PX))
    builder = MosaicBuilder(tile_manager, grid_size, color_mode, strength, match_space,
                            detail_weight, quantize_k, long_edge)
    return builder.build(image)
