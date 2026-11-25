"""Image loading, preprocessing and grid division.

The steps that come before tile matching:

1. `load_image`            -- read a file, or accept an array or PIL image, as RGB uint8
2. `to_working_resolution` -- resize once, so the longest side is `long_edge` pixels
3. `crop_to_grid`          -- center-crop so the image splits into whole square cells
4. `cell_statistics`       -- mean color (and optionally contrast) of every cell

`prepare_image` runs steps 1-3 in one call. `segmented_view` turns the cell colors back
into a full-size picture of the grid, for display.

Example:
    >>> img = np.zeros((300, 400, 3), np.uint8)
    >>> grid = prepare_image(img, grid_size=20, long_edge=400)
    >>> grid.n_rows, grid.n_cols, grid.cell_px
    (15, 20, 20)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from . import config
from .config import LUMA
from .utils import MosaicInputError, check_int, read_image_file


class GridImage(NamedTuple):
    """An image cropped to a whole number of square cells.

    Attributes
    ----------
    image : np.ndarray
        (n_rows * cell_px, n_cols * cell_px, 3) uint8 RGB image.
    n_rows : int
        Number of cell rows.
    n_cols : int
        Number of cell columns.
    cell_px : int
        Side of every cell, in pixels.
    """

    image: np.ndarray
    n_rows: int
    n_cols: int
    cell_px: int


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def as_rgb_uint8(image) -> np.ndarray:
    """Convert any image array to a contiguous (H, W, 3) uint8 RGB array.

    Handles the formats a web upload or an image library can produce: grayscale,
    single-channel, RGBA, and float images in [0, 1] or [0, 255].

    Parameters
    ----------
    image : array-like
        (H, W), (H, W, 1), (H, W, 3) or (H, W, 4) image of any numeric dtype, or an
        object NumPy can turn into one (e.g. a PIL image).

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB image.

    Raises
    ------
    MosaicInputError
        If there is no image, the array has the wrong shape, is empty, or contains
        NaN or infinite values.

    Example
    -------
    >>> as_rgb_uint8(np.full((2, 2), 0.5)).tolist()[0][0]
    [127, 127, 127]
    """
    if image is None:
        raise MosaicInputError("no image was given")
    arr = np.asarray(image)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        raise MosaicInputError(f"image must be numeric pixel data, got dtype {arr.dtype}")
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise MosaicInputError(
            f"image must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise MosaicInputError(f"image is empty (shape {arr.shape})")

    if arr.dtype != np.uint8:
        if not np.all(np.isfinite(arr)):
            raise MosaicInputError("image contains NaN or infinite values")
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0                     # floats from image libraries are often 0-1
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(arr)


def load_image(source) -> np.ndarray:
    """Load an image from a file path, a NumPy array or a PIL image, as RGB uint8.

    Parameters
    ----------
    source : str, Path, np.ndarray or PIL.Image.Image
        A path to an image file, or the image itself.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB image.

    Raises
    ------
    MosaicInputError
        If the file cannot be read or the image data cannot be used
        (see `utils.read_image_file` and `as_rgb_uint8`).

    Example
    -------
    >>> load_image(np.zeros((4, 5), np.uint8)).shape
    (4, 5, 3)
    """
    if isinstance(source, (str, Path)):
        return read_image_file(source)
    return as_rgb_uint8(source)


# --------------------------------------------------------------------------------------
# Resizing and grid division
# --------------------------------------------------------------------------------------


def to_working_resolution(image: np.ndarray, long_edge: int = config.DEFAULT_LONG_EDGE) -> np.ndarray:
    """Resize an image once, so that its longest side is `long_edge` pixels.

    A fixed working resolution keeps the cost of a mosaic independent of how large a
    photo the user uploads. Shrinking uses area averaging, which averages over each
    output pixel's footprint and so keeps the cell colors accurate; enlarging uses cubic
    interpolation. An image that already has the right size is returned unchanged.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 RGB image.
    long_edge : int, default config.DEFAULT_LONG_EDGE
        Target length of the longest side, between `config.MIN_LONG_EDGE` and
        `config.MAX_LONG_EDGE`.

    Returns
    -------
    np.ndarray
        The resized (or unchanged) image; the aspect ratio is kept.

    Raises
    ------
    MosaicInputError
        If `long_edge` is out of range.

    Example
    -------
    >>> to_working_resolution(np.zeros((100, 200, 3), np.uint8), 400).shape
    (200, 400, 3)
    """
    long_edge = check_int("long_edge", long_edge, config.MIN_LONG_EDGE, config.MAX_LONG_EDGE)
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if np.isclose(scale, 1.0):
        return image
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def grid_shape(height: int, width: int, grid_size: int) -> tuple[int, int, int]:
    """Work out the grid for an image: square cells, about `grid_size` of them across.

    The requested count sets the cell size (`width // grid_size`); the grid then takes
    every whole cell that fits on each axis. This keeps the crop below one cell per axis.

    Parameters
    ----------
    height, width : int
        Image size in pixels.
    grid_size : int
        Requested number of cells across, between `config.MIN_GRID_SIZE` and
        `config.MAX_GRID_SIZE`.

    Returns
    -------
    tuple of (int, int, int)
        (n_rows, n_cols, cell_px).

    Raises
    ------
    MosaicInputError
        If `grid_size` is out of range, the image is narrower than `grid_size` pixels,
        shorter than one cell, or the cells would be larger than `config.MAX_CELL_PX`.

    Example
    -------
    >>> grid_shape(512, 512, 32)
    (32, 32, 16)
    """
    grid_size = check_int("grid_size", grid_size, config.MIN_GRID_SIZE, config.MAX_GRID_SIZE)
    if width < grid_size:
        raise MosaicInputError(
            f"the image is {width} px wide, too small for a grid of {grid_size} cells across "
            f"(each cell needs at least 1 px); use a smaller grid or a larger image")
    cell = width // grid_size
    if cell > config.MAX_CELL_PX:
        raise MosaicInputError(
            f"cells would be {cell} px wide, more than the {config.MAX_CELL_PX} px limit; "
            f"use a larger grid size or a smaller working resolution")
    if height < cell:
        raise MosaicInputError(
            f"the image is {height} px tall, shorter than one {cell} px cell; "
            f"use a larger grid size or a less wide image")
    return height // cell, width // cell, cell


def crop_to_grid(image: np.ndarray, grid_size: int) -> GridImage:
    """Center-crop an image so it splits exactly into square cells.

    At most `cell_px - 1` pixels are removed per axis, so the grid can be handled as a
    plain reshape, with no padding and no partial cells at the edges.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) uint8 RGB image.
    grid_size : int
        Requested number of cells across (see `grid_shape`).

    Returns
    -------
    GridImage
        The cropped, contiguous image and its grid layout.

    Raises
    ------
    MosaicInputError
        If the grid does not fit the image (see `grid_shape`).

    Example
    -------
    >>> crop_to_grid(np.zeros((100, 130, 3), np.uint8), 13).image.shape
    (100, 130, 3)
    """
    h, w = image.shape[:2]
    n_rows, n_cols, cell = grid_shape(h, w, grid_size)
    target_h, target_w = n_rows * cell, n_cols * cell
    top, left = (h - target_h) // 2, (w - target_w) // 2
    cropped = np.ascontiguousarray(image[top:top + target_h, left:left + target_w])
    return GridImage(cropped, n_rows, n_cols, cell)


def prepare_image(source, grid_size: int = config.DEFAULT_GRID_SIZE,
                  long_edge: int = config.DEFAULT_LONG_EDGE) -> GridImage:
    """Load, resize and crop an image, ready for the mosaic (steps 1-3 in one call).

    Parameters
    ----------
    source : str, Path, np.ndarray or PIL.Image.Image
        The input image or a path to it.
    grid_size : int, default config.DEFAULT_GRID_SIZE
        Requested number of cells across.
    long_edge : int, default config.DEFAULT_LONG_EDGE
        Working resolution: the longest side after resizing, in pixels.

    Returns
    -------
    GridImage
        The cropped image and its grid layout.

    Raises
    ------
    MosaicInputError
        If the image cannot be loaded or the parameters are out of range.

    Example
    -------
    >>> prepare_image(np.zeros((512, 512, 3), np.uint8), 32, 512)[1:]
    (32, 32, 16)
    """
    return crop_to_grid(to_working_resolution(load_image(source), long_edge), grid_size)


# --------------------------------------------------------------------------------------
# Cell statistics
# --------------------------------------------------------------------------------------


def cell_statistics(image: np.ndarray, n_rows: int, n_cols: int, cell_px: int,
                    with_std: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean color, and optionally brightness contrast, of every grid cell at once.

    Profiling the Lab 1 loop showed that computing the mean one cell at a time took 24%
    of its run time. Here the mean of all cells comes from two passes over contiguous
    memory (see `cell_means`). The contrast is only needed for texture-aware matching,
    so it is computed only when asked for.

    Parameters
    ----------
    image : np.ndarray
        (n_rows * cell_px, n_cols * cell_px, 3) uint8 RGB image (see `crop_to_grid`).
    n_rows, n_cols : int
        Grid layout.
    cell_px : int
        Side of every cell, in pixels.
    with_std : bool, default False
        Also return the standard deviation of each cell's brightness.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        (n_rows, n_cols, 3) float32 mean colors, and (n_rows, n_cols) float32
        brightness standard deviations (None when `with_std` is False).

    Example
    -------
    >>> mean, std = cell_statistics(np.full((4, 4, 3), 7, np.uint8), 2, 2, 2, with_std=True)
    >>> float(mean[0, 0, 0]), float(std[0, 0])
    (7.0, 0.0)
    """
    if not with_std:
        return cell_means(image, n_rows, n_cols, cell_px), None

    blocks = image.reshape(n_rows, cell_px, n_cols, cell_px, 3).astype(np.float32)
    mean = blocks.mean(axis=(1, 3))
    std = (blocks @ LUMA).std(axis=(1, 3))
    return mean, std.astype(np.float32)


def cell_means(image: np.ndarray, n_rows: int, n_cols: int, cell_px: int) -> np.ndarray:
    """Mean color of every grid cell, in two passes over contiguous memory.

    1. Viewed as an (H * n_cols, cell_px, 3) image, every row is one cell's slice of one
       pixel row. `cv2.reduce` sums each row in one optimized pass, straight from uint8.
    2. The `cell_px` partial sums of each cell are added with NumPy. This array is
       `cell_px * 3` times smaller than the image, so this step is cheap.

    This is about 25 times faster than `reshape(...).mean(axis=(1, 3))`, which NumPy
    handles slowly because the two reduced axes are not next to each other in memory.
    The sums are whole numbers below 2**24 (cells of at most `config.MAX_CELL_PX` px),
    which float32 holds exactly, so the result is identical to that slower version.

    Parameters
    ----------
    image : np.ndarray
        (n_rows * cell_px, n_cols * cell_px, 3) uint8 RGB image.
    n_rows, n_cols : int
        Grid layout.
    cell_px : int
        Side of every cell, in pixels.

    Returns
    -------
    np.ndarray
        (n_rows, n_cols, 3) float32 mean colors.

    Example
    -------
    >>> cell_means(np.full((4, 6, 3), 9, np.uint8), 2, 3, 2)[1, 2].tolist()
    [9.0, 9.0, 9.0]
    """
    flat = np.ascontiguousarray(image).reshape(n_rows * cell_px * n_cols, cell_px, 3)
    row_sums = cv2.reduce(flat, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)   # (H * n_cols, 1, 3)
    sums = row_sums.reshape(n_rows, cell_px, n_cols, 3).sum(axis=1)
    return sums / np.float32(cell_px * cell_px)


def quantize_colors(colors: np.ndarray, k: int = config.DEFAULT_PALETTE_K) -> np.ndarray:
    """Reduce an array of colors to at most `k` distinct colors with k-means.

    Used on the cell colors (not the full image), which gives the same flatter, poster-
    like palette for a fraction of the cost. k-means starts from random centers; call
    `cv2.setRNGSeed` first for repeatable results.

    Parameters
    ----------
    colors : np.ndarray
        (..., 3) colors in [0, 255].
    k : int, default config.DEFAULT_PALETTE_K
        Maximum number of colors, between 1 and `config.MAX_QUANTIZE_K`. 1 or less
        returns the colors unchanged.

    Returns
    -------
    np.ndarray
        uint8 array with the same shape as `colors`.

    Raises
    ------
    MosaicInputError
        If `k` is out of range.

    Example
    -------
    >>> out = quantize_colors(np.random.default_rng(0).integers(0, 256, (10, 10, 3)), k=4)
    >>> len(np.unique(out.reshape(-1, 3), axis=0)) <= 4
    True
    """
    k = check_int("quantize_k", k, 0, config.MAX_QUANTIZE_K)
    colors = np.asarray(colors)
    if k <= 1:
        return colors.astype(np.uint8)

    samples = colors.reshape(-1, 3).astype(np.float32)
    k = min(k, len(np.unique(samples, axis=0)))
    if k <= 1:
        return colors.astype(np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, attempts=3,
                                    flags=cv2.KMEANS_PP_CENTERS)
    return centers[labels.ravel()].reshape(colors.shape).clip(0, 255).astype(np.uint8)


def segmented_view(cell_mean: np.ndarray, height: int, width: int) -> np.ndarray:
    """Draw the grid itself: every cell filled with its mean color, at full size.

    Parameters
    ----------
    cell_mean : np.ndarray
        (n_rows, n_cols, 3) cell colors in [0, 255].
    height, width : int
        Output size in pixels (normally the cropped image's size).

    Returns
    -------
    np.ndarray
        (height, width, 3) uint8 image.

    Example
    -------
    >>> segmented_view(np.full((2, 2, 3), 50.0, np.float32), 8, 8).shape
    (8, 8, 3)
    """
    cells = np.clip(cell_mean, 0, 255).astype(np.uint8)
    return cv2.resize(cells, (width, height), interpolation=cv2.INTER_NEAREST)
