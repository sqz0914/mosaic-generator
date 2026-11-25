"""Tile sets: creating or loading tiles, caching them, and extracting their features.

A tile set is a stack of square RGB tiles plus one "key" color per tile (the tile's mean
color). Matching a grid cell to a tile is then a nearest-neighbor search against the keys.

Tiles come from one of two sources:

* **Built-in sets**, drawn by code (`TILE_SETS`). They need no image files, and they can
  be drawn at any resolution.
* **A folder of images** (`TileManager(tile_directory=...)`). Every readable image in the
  folder becomes one tile. Unreadable files are skipped with a warning.

`TileManager` caches everything that depends only on the tiles -- the tiles at each
size, their matching keys, their texture, and the tile part of the recolor -- so a
mosaic never recomputes per-tile work. Profiling the Lab 1 loop showed that this
repeated work (`tile @ LUMA` for every cell) took 16% of its time.

Example:
    >>> tm = TileManager("Halftone Dots", tile_px=16)
    >>> tm.n_tiles, tm.mode
    (14, 'intensity')
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config
from .config import LUMA
from .utils import MosaicInputError, check_choice, check_int, freeze, read_image_file


# --------------------------------------------------------------------------------------
# Tile set container
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSet:
    """A stack of square tiles and the color each one stands for.

    Attributes
    ----------
    name : str
        Display name of the set.
    tiles : np.ndarray
        (N, T, T, 3) uint8 tile images, sorted from darkest to brightest.
    keys : np.ndarray
        (N, 3) float32 mean RGB color of each tile; cells are matched against these.
    mode : str
        "intensity": cells are matched on brightness only (monochrome tile art).
        "color": cells are matched on the full color.
    description : str
        One-line description for the web app.
    """

    name: str
    tiles: np.ndarray
    keys: np.ndarray
    mode: str
    description: str = ""

    @property
    def tile_size(self) -> int:
        """Side length of every tile.

        Returns
        -------
        int
            Tile side T in pixels.
        """
        return int(self.tiles.shape[1])

    @property
    def n_tiles(self) -> int:
        """Number of tiles in the set.

        Returns
        -------
        int
            Number of tiles N.
        """
        return int(self.tiles.shape[0])

    @property
    def luma_keys(self) -> np.ndarray:
        """Brightness of each tile key.

        Returns
        -------
        np.ndarray
            (N,) float32 brightness, increasing (the tiles are sorted by brightness).
        """
        return self.keys @ LUMA


def make_tile_set(name: str, tiles: np.ndarray, mode: str, description: str = "") -> TileSet:
    """Turn a stack of tile images into a `TileSet` with measured keys.

    Each key is the tile's own mean color, so the tile chosen for a cell keeps that
    cell's average color as closely as the set allows. Tiles are sorted by brightness, so
    previews read as a ramp from dark to light.

    Parameters
    ----------
    name : str
        Display name.
    tiles : np.ndarray
        (N, T, T, 3) tile images, any numeric dtype in [0, 255].
    mode : str
        "intensity" or "color".
    description : str, optional
        One-line description.

    Returns
    -------
    TileSet
        A set whose `tiles` and `keys` are read-only.

    Example
    -------
    >>> ts = make_tile_set("two", np.stack([np.zeros((4, 4, 3)), np.full((4, 4, 3), 255)]), "intensity")
    >>> ts.keys[:, 0].tolist()
    [0.0, 255.0]
    """
    tiles = np.ascontiguousarray(np.clip(tiles, 0, 255).astype(np.uint8))
    keys = tiles.reshape(len(tiles), -1, 3).mean(axis=1).astype(np.float32)
    order = np.argsort(keys @ LUMA, kind="stable")
    return TileSet(name, freeze(tiles[order]), freeze(keys[order]), mode, description)


# --------------------------------------------------------------------------------------
# Built-in tile sets, drawn by code
# --------------------------------------------------------------------------------------


def _render_px(size: int) -> int:
    """Resolution at which a tile is drawn before it is shrunk to its final size.

    Drawing large and area-averaging down gives much smoother edges than drawing small.

    Parameters
    ----------
    size : int
        Final tile side in pixels.

    Returns
    -------
    int
        Drawing side in pixels: `size * config.TILE_RENDER_SCALE`, but at least
        `config.TILE_RENDER_MIN_PX`.
    """
    return max(config.TILE_RENDER_MIN_PX, size * config.TILE_RENDER_SCALE)


def _shrink(tiles: list[np.ndarray], size: int) -> np.ndarray:
    """Area-average drawn tiles down to their final size and stack them.

    Parameters
    ----------
    tiles : list of np.ndarray
        Square (hi, hi, 3) uint8 drawings.
    size : int
        Final tile side in pixels.

    Returns
    -------
    np.ndarray
        (N, size, size, 3) uint8.
    """
    return np.stack([cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles])


def _canvas(hi: int, value: int = 0) -> np.ndarray:
    """A blank square drawing surface.

    Parameters
    ----------
    hi : int
        Side in pixels.
    value : int, default 0
        Gray level to fill with (0 = black, 255 = white).

    Returns
    -------
    np.ndarray
        (hi, hi, 3) uint8 filled with `value`.
    """
    return np.full((hi, hi, 3), value, dtype=np.uint8)


ASCII_RAMP = " .:-=+ic*#%@"

# (ink, paper) gray levels. A stroke font inks only about a quarter of its cell, so one
# ink/paper pair spans barely 65 brightness levels. Four pairs -- light ink on dark paper,
# then dark ink on light paper -- cover the full 0-255 range with no gap wider than ~25.
ASCII_PHASES = ((255, 0), (255, 88), (0, 168), (0, 255))


def _render_glyph(ch: str, hi: int, ink: int, paper: int) -> np.ndarray:
    """Draw one centered character.

    Parameters
    ----------
    ch : str
        The character; a space gives an empty tile.
    hi : int
        Drawing side in pixels.
    ink : int
        Gray level of the character.
    paper : int
        Gray level of the background.

    Returns
    -------
    np.ndarray
        (hi, hi, 3) uint8 drawing.
    """
    tile = _canvas(hi, paper)
    if ch == " ":
        return tile
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, hi / 30.0, max(1, int(hi / 24))
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    cv2.putText(tile, ch, ((hi - tw) // 2, (hi + th) // 2), font, scale, (ink,) * 3,
                thickness, cv2.LINE_AA)
    return tile


def _ascii_tiles(size: int) -> np.ndarray:
    """Draw the "ASCII Art" set: a character-density ramp over four ink/paper pairs.

    Parameters
    ----------
    size : int
        Tile side in pixels.

    Returns
    -------
    np.ndarray
        (48, size, size, 3) uint8.
    """
    hi = _render_px(size)
    return _shrink([_render_glyph(ch, hi, ink, paper)
                    for ink, paper in ASCII_PHASES for ch in ASCII_RAMP], size)


def _halftone_tiles(size: int, n: int = 14) -> np.ndarray:
    """Draw the "Halftone Dots" set: a centered dot whose area grows with brightness.

    The radius grows with the square root of the level, so the dot area -- and with it
    the mean brightness -- increases evenly across the set.

    Parameters
    ----------
    size : int
        Tile side in pixels.
    n : int, default 14
        Number of tiles.

    Returns
    -------
    np.ndarray
        (n, size, size, 3) uint8.
    """
    hi = _render_px(size)
    max_r = hi * 0.72                       # big enough for the last dot to cover the tile
    tiles = []
    for i in range(n):                      # runs when the set is first built, then cached
        tile = _canvas(hi)
        r = int(round(max_r * np.sqrt(i / (n - 1))))
        if r > 0:
            cv2.circle(tile, (hi // 2, hi // 2), r, (255, 255, 255), -1, cv2.LINE_AA)
        tiles.append(tile)
    return _shrink(tiles, size)


def _geometric_tiles(size: int) -> np.ndarray:
    """Draw the "Geometric Blocks" set: bars, wedges, a diamond and squares.

    The shapes cover a range of ink coverage; the last two are dark marks on white, to
    fill the bright end of the ramp.

    Parameters
    ----------
    size : int
        Tile side in pixels.

    Returns
    -------
    np.ndarray
        (15, size, size, 3) uint8.
    """
    hi = _render_px(size)
    white, black = (255, 255, 255), (0, 0, 0)

    shapes = [_canvas(hi)]                                         # empty
    for frac in (0.25, 0.45):                                      # square outlines
        t, m = _canvas(hi), int(hi * (0.5 - frac / 2))
        cv2.rectangle(t, (m, m), (hi - m, hi - m), white, max(1, hi // 12))
        shapes.append(t)
    for pts in ((0, 0, hi, hi), (hi, 0, 0, hi)):                   # diagonal bars
        t = _canvas(hi)
        cv2.line(t, pts[:2], pts[2:], white, max(2, hi // 6), cv2.LINE_AA)
        shapes.append(t)
    for tri in ([(0, 0), (hi, 0), (0, hi)], [(hi, 0), (hi, hi), (0, hi)]):   # wedges
        t = _canvas(hi)
        cv2.fillPoly(t, [np.array(tri, dtype=np.int32)], white)
        shapes.append(t)
    t, c = _canvas(hi), hi // 2                                    # diamond
    cv2.fillPoly(t, [np.array([(c, 0), (hi, c), (c, hi), (0, c)], dtype=np.int32)], white)
    shapes.append(t)
    for frac in (0.4, 0.55, 0.7, 0.85, 1.0):                       # growing squares
        t, m = _canvas(hi), int(hi * (1 - frac) / 2)
        cv2.rectangle(t, (m, m), (hi - m - 1, hi - m - 1), white, -1)
        shapes.append(t)
    for frac in (0.55, 0.35):                                      # dark squares on white
        t, m = _canvas(hi, 255), int(hi * (1 - frac) / 2)
        cv2.rectangle(t, (m, m), (hi - m - 1, hi - m - 1), black, -1)
        shapes.append(t)
    return _shrink(shapes, size)


LEGO_PALETTE = [
    (13, 13, 13), (35, 42, 110), (0, 85, 191), (108, 192, 227),
    (0, 133, 43), (150, 191, 77), (255, 205, 3), (245, 125, 32),
    (180, 0, 0), (255, 255, 255), (161, 165, 162), (99, 95, 98),
    (149, 185, 11), (172, 120, 186), (231, 172, 200), (88, 57, 39),
]


def _lego_tiles(size: int) -> np.ndarray:
    """Draw the "LEGO Studs" set: one studded plate per palette color.

    Each stud has a darker rim and a lighter face, so it still reads as 3D when the tile
    is only a few pixels wide.

    Parameters
    ----------
    size : int
        Tile side in pixels.

    Returns
    -------
    np.ndarray
        (16, size, size, 3) uint8.
    """
    hi = _render_px(size)
    c, r = hi // 2, int(hi * 0.30)
    tiles = []
    for base in LEGO_PALETTE:               # runs when the set is first built, then cached
        t = _canvas(hi)
        t[:] = base
        shade = np.array(base, dtype=np.float32)
        rim = tuple(np.clip(shade * 0.75, 0, 255).astype(int).tolist())
        face = tuple(np.clip(shade * 1.25 + 12, 0, 255).astype(int).tolist())
        cv2.circle(t, (c, c), r, rim, -1, cv2.LINE_AA)
        cv2.circle(t, (c, c - hi // 40), r - max(1, hi // 32), face, -1, cv2.LINE_AA)
        tiles.append(t)
    return _shrink(tiles, size)


# name -> (generator, mode, description)
_BUILT_IN: dict[str, tuple[Callable[[int], np.ndarray], str, str]] = {
    "ASCII Art": (_ascii_tiles, "intensity", "Character density ramp, light on dark."),
    "Halftone Dots": (_halftone_tiles, "intensity", "Newspaper screen; dot area tracks brightness."),
    "Geometric Blocks": (_geometric_tiles, "intensity", "Bars, wedges and diamonds across a coverage ramp."),
    "LEGO Studs": (_lego_tiles, "color", "Studded plates in a fixed brick palette; matched on color."),
}

TILE_SETS: list[str] = list(_BUILT_IN)


@lru_cache(maxsize=config.TILE_CACHE_SIZE)
def _built_in_tile_set(name: str, tile_px: int) -> TileSet:
    """Draw a built-in set once; later calls with the same arguments return the cached set.

    The cache is shared by all `TileManager` objects.

    Parameters
    ----------
    name : str
        A name from `TILE_SETS`.
    tile_px : int
        Tile side in pixels.

    Returns
    -------
    TileSet
        The drawn set.
    """
    generator, mode, description = _BUILT_IN[name]
    return make_tile_set(name, generator(tile_px), mode, description)


# --------------------------------------------------------------------------------------
# Tiles from a folder of images
# --------------------------------------------------------------------------------------


def _center_square(img: np.ndarray) -> np.ndarray:
    """Crop the largest centered square out of an image.

    Parameters
    ----------
    img : np.ndarray
        (H, W, 3) image.

    Returns
    -------
    np.ndarray
        (S, S, 3) contiguous copy, S = min(H, W).
    """
    h, w = img.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return np.ascontiguousarray(img[top:top + side, left:left + side])


def load_tile_directory(directory: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """Read every image in a folder as a square tile source.

    Each image is center-cropped to a square. Files that cannot be read are skipped with
    a `UserWarning` that names them, so one broken file does not stop the mosaic.

    Parameters
    ----------
    directory : str or Path
        Folder that contains the tile images (not searched recursively).

    Returns
    -------
    tuple of (list of np.ndarray, list of str)
        The square (S, S, 3) uint8 RGB images, sorted by file name, and the names of
        the files that could not be read.

    Raises
    ------
    MosaicInputError
        If the folder does not exist, contains no image files, or fewer than two of them
        can be read.

    Example
    -------
    >>> images, skipped = load_tile_directory("my_tiles/")  # doctest: +SKIP
    """
    folder = Path(directory)
    if not folder.exists():
        raise MosaicInputError(f"tile directory not found: {folder}")
    if not folder.is_dir():
        raise MosaicInputError(f"tile directory is not a folder: {folder}")

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if not files:
        allowed = ", ".join(config.IMAGE_EXTENSIONS)
        raise MosaicInputError(f"no tile images ({allowed}) found in {folder}")

    images, skipped = [], []
    for path in files:                      # runs once per tile file, when the folder is loaded
        try:
            images.append(_center_square(read_image_file(path)))
        except MosaicInputError as err:
            skipped.append(path.name)
            warnings.warn(f"skipping tile {path.name}: {err}", UserWarning, stacklevel=2)

    if len(images) < 2:
        raise MosaicInputError(
            f"need at least 2 readable tile images in {folder}, found {len(images)}"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
        )
    return images, skipped


# --------------------------------------------------------------------------------------
# Feature extraction (shared by tile keys and grid cells)
# --------------------------------------------------------------------------------------


def to_match_space(rgb: np.ndarray, space: str) -> np.ndarray:
    """Convert RGB colors to float32 in the matching color space.

    Parameters
    ----------
    rgb : np.ndarray
        (..., 3) colors in [0, 255].
    space : str
        "RGB" (returned as float32) or "CIE-Lab".

    Returns
    -------
    np.ndarray
        (..., 3) float32 colors in `space`.

    Example
    -------
    >>> float(to_match_space(np.array([[255, 255, 255]]), "CIE-Lab").round()[0, 0])
    100.0
    """
    if space != "CIE-Lab":
        return np.asarray(rgb, dtype=np.float32)
    flat = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).reshape(1, -1, 3) / 255.0)
    return cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).reshape(np.shape(rgb))


def color_features(rgb: np.ndarray, mode: str, space: str) -> np.ndarray:
    """Matching features of RGB colors, used for both tile keys and grid cells.

    Intensity sets are matched on brightness only (one feature); color sets on the full
    color (three features). CIE-Lab lightness L* is on [0, 100], so it is scaled by 2.55
    to stay comparable with RGB brightness.

    Parameters
    ----------
    rgb : np.ndarray
        (..., 3) colors in [0, 255].
    mode : str
        "intensity" or "color" (the tile set's mode).
    space : str
        "RGB" or "CIE-Lab".

    Returns
    -------
    np.ndarray
        (..., F) float32 features, F = 1 for intensity and 3 for color.

    Example
    -------
    >>> color_features(np.array([[255.0, 255.0, 255.0]]), "intensity", "RGB").round(1)
    array([[255.]], dtype=float32)
    """
    if mode == "intensity":
        if space == "CIE-Lab":
            return to_match_space(rgb, space)[..., :1] * 2.55
        return (np.asarray(rgb, dtype=np.float32) @ LUMA)[..., None]
    return to_match_space(rgb, space)


# --------------------------------------------------------------------------------------
# TileManager
# --------------------------------------------------------------------------------------


class TileManager:
    """Provide one tile set at any resolution, with its features cached.

    Parameters
    ----------
    tile_set : str, optional
        Name of a built-in set (see `TILE_SETS`). Defaults to `config.DEFAULT_TILE_SET`
        when no `tile_directory` is given.
    tile_px : int, default config.DEFAULT_TILE_PX
        Default tile resolution in pixels, between `config.MIN_TILE_PX` and
        `config.MAX_TILE_PX`. Methods accept other sizes too, which are cached separately.
    tile_directory : str or Path, optional
        Folder of tile images to use instead of a built-in set. Its tiles are matched on
        full color.

    Attributes
    ----------
    name : str
        Display name of the set.
    mode : str
        "intensity" or "color".
    description : str
        One-line description.
    n_tiles : int
        Number of tiles.
    skipped_files : list of str
        Tile files that could not be read (always empty for built-in sets).

    Raises
    ------
    MosaicInputError
        If both `tile_set` and `tile_directory` are given, the set name is unknown,
        `tile_px` is out of range, or the folder cannot be used (see
        `load_tile_directory`).

    Example
    -------
    >>> tm = TileManager("LEGO Studs", tile_px=8)
    >>> tm.tiles().tiles.shape
    (16, 8, 8, 3)
    """

    def __init__(self, tile_set: str | None = None, tile_px: int = config.DEFAULT_TILE_PX,
                 tile_directory: str | Path | None = None) -> None:
        """Validate the arguments and prepare the tile source.

        Parameters
        ----------
        tile_set : str, optional
            Name of a built-in set.
        tile_px : int, default config.DEFAULT_TILE_PX
            Default tile resolution in pixels.
        tile_directory : str or Path, optional
            Folder of tile images.

        Returns
        -------
        None
        """
        if tile_set is not None and tile_directory is not None:
            raise MosaicInputError("give either tile_set or tile_directory, not both")

        self.tile_px = check_int("tile_px", tile_px, config.MIN_TILE_PX, config.MAX_TILE_PX)
        self._cache: dict[tuple, np.ndarray | TileSet] = {}
        self._lock = threading.Lock()

        if tile_directory is None:
            self.name = check_choice("tile_set", tile_set or config.DEFAULT_TILE_SET, TILE_SETS)
            self.mode = _BUILT_IN[self.name][1]
            self.description = _BUILT_IN[self.name][2]
            self.skipped_files: list[str] = []
            self._sources: list[np.ndarray] | None = None
        else:
            self._sources, self.skipped_files = load_tile_directory(tile_directory)
            self.name = f"Folder: {Path(tile_directory).name}"
            self.mode = "color"
            self.description = f"{len(self._sources)} tiles loaded from {Path(tile_directory)}."

        self.n_tiles = self.tiles().n_tiles

    def __repr__(self) -> str:
        """Short text form for printing and debugging.

        Returns
        -------
        str
            E.g. "TileManager(name='Halftone Dots', tile_px=16, n_tiles=14)".
        """
        return f"TileManager(name={self.name!r}, tile_px={self.tile_px}, n_tiles={self.n_tiles})"

    @staticmethod
    def available_tile_sets() -> list[str]:
        """Names of the built-in tile sets.

        Returns
        -------
        list of str
            E.g. ["ASCII Art", "Halftone Dots", "Geometric Blocks", "LEGO Studs"].

        Example
        -------
        >>> "Halftone Dots" in TileManager.available_tile_sets()
        True
        """
        return list(TILE_SETS)

    # ---------------------------------------------------------------- caching helpers

    def _cached(self, key: tuple, compute: Callable[[], np.ndarray | TileSet]) -> np.ndarray | TileSet:
        """Return the cached value for `key`, computing and storing it on first use.

        Parameters
        ----------
        key : tuple
            Cache key, e.g. ("base", 16, "Modulate", 0.85).
        compute : callable
            Function with no arguments that builds the value.

        Returns
        -------
        np.ndarray or TileSet
            The cached value. If two threads compute it at the same time, both get the
            one that was stored first.
        """
        value = self._cache.get(key)
        if value is None:
            value = compute()
            with self._lock:
                value = self._cache.setdefault(key, value)
        return value

    def _size(self, tile_px: int | None) -> int:
        """Validate an optional tile size, defaulting to the manager's `tile_px`.

        Sizes below `config.MIN_TILE_PX` are allowed here: very fine grids shrink their
        tiles to fit the output cap, and scoring renders one tile pixel per cell pixel.

        Parameters
        ----------
        tile_px : int or None
            Requested tile side in pixels.

        Returns
        -------
        int
            The size to use, between 1 and `config.MAX_CELL_PX`.
        """
        if tile_px is None:
            return self.tile_px
        return check_int("tile_px", tile_px, 1, config.MAX_CELL_PX)

    # ---------------------------------------------------------------- computations

    def _resize_folder_tiles(self, size: int) -> TileSet:
        """Resize the loaded folder images to one tile size.

        Shrinking uses area averaging; enlarging (tiny source images) uses cubic
        interpolation.

        Parameters
        ----------
        size : int
            Tile side in pixels.

        Returns
        -------
        TileSet
            The folder's tiles at `size`.
        """
        resized = [cv2.resize(src, (size, size),
                              interpolation=cv2.INTER_AREA if src.shape[0] >= size else cv2.INTER_CUBIC)
                   for src in self._sources]
        return make_tile_set(self.name, np.stack(resized), self.mode, self.description)

    def _compute_keys(self, size: int, space: str) -> np.ndarray:
        """Matching features of the tile keys, uncached (see `match_keys`).

        Parameters
        ----------
        size : int
            Tile side in pixels.
        space : str
            "RGB" or "CIE-Lab".

        Returns
        -------
        np.ndarray
            (N, F) float32, read-only.
        """
        return freeze(color_features(self.tiles(size).keys, self.mode, space))

    def _compute_detail(self, size: int) -> np.ndarray:
        """Internal contrast of each tile, uncached (see `tile_detail`).

        Parameters
        ----------
        size : int
            Tile side in pixels.

        Returns
        -------
        np.ndarray
            (N,) float32 in [0, 1], read-only.
        """
        luma = self.tiles(size).tiles.astype(np.float32) @ LUMA
        return freeze(np.clip(luma.reshape(len(luma), -1).std(axis=1) / 96.0, 0, 1))

    def _compute_base(self, size: int, color_mode: str, strength: float) -> np.ndarray:
        """Tile part of the recolor, uncached (see `render_base`).

        Parameters
        ----------
        size : int
            Tile side in pixels.
        color_mode : str
            One of `config.COLOR_MODES`.
        strength : float
            Recolor strength in [0, 1].

        Returns
        -------
        np.ndarray
            (N, T, T, 3) uint8, read-only.
        """
        tiles = self.tiles(size).tiles
        if color_mode == "Tile only" or strength <= 0:
            return tiles                                  # already uint8 and read-only
        tiles = tiles.astype(np.float32)
        if color_mode == "Blend":
            base = tiles * (1.0 - strength)
        else:
            base = tiles * (1.0 - strength) + (tiles @ LUMA)[..., None] * strength
        return freeze(np.rint(np.clip(base, 0, 255)).astype(np.uint8))

    # ---------------------------------------------------------------- public API

    def tiles(self, tile_px: int | None = None) -> TileSet:
        """The tile set at a given resolution (drawn or resized once, then cached).

        Parameters
        ----------
        tile_px : int, optional
            Tile side in pixels. Defaults to the manager's `tile_px`.

        Returns
        -------
        TileSet
            Tiles of shape (N, tile_px, tile_px, 3), sorted by brightness.

        Example
        -------
        >>> TileManager("Halftone Dots").tiles(8).tile_size
        8
        """
        size = self._size(tile_px)
        if self._sources is None:
            return _built_in_tile_set(self.name, size)
        return self._cached(("tiles", size), partial(self._resize_folder_tiles, size))

    def match_keys(self, space: str = config.DEFAULT_MATCH_SPACE,
                   tile_px: int | None = None) -> np.ndarray:
        """Matching features of every tile key (computed once, then cached).

        Parameters
        ----------
        space : str, default "RGB"
            "RGB" or "CIE-Lab".
        tile_px : int, optional
            Resolution whose keys to use. Defaults to the manager's `tile_px`.

        Returns
        -------
        np.ndarray
            (N, F) float32, read-only. F = 1 for intensity sets, 3 for color sets.

        Example
        -------
        >>> TileManager("Halftone Dots").match_keys("RGB").shape
        (14, 1)
        """
        space = check_choice("match_space", space, config.MATCH_SPACES)
        size = self._size(tile_px)
        return self._cached(("keys", size, space), partial(self._compute_keys, size, space))

    def tile_detail(self, tile_px: int | None = None) -> np.ndarray:
        """Internal contrast of every tile, for texture-aware matching (cached).

        Parameters
        ----------
        tile_px : int, optional
            Resolution to measure at. Defaults to the manager's `tile_px`.

        Returns
        -------
        np.ndarray
            (N,) float32 in [0, 1], read-only: the standard deviation of each tile's
            brightness, divided by 96 and clipped.

        Example
        -------
        >>> float(TileManager("Halftone Dots").tile_detail()[0])
        0.0
        """
        size = self._size(tile_px)
        return self._cached(("detail", size), partial(self._compute_detail, size))

    def render_base(self, tile_px: int | None = None, color_mode: str = config.DEFAULT_COLOR_MODE,
                    strength: float = config.DEFAULT_STRENGTH) -> np.ndarray:
        """The part of every recolored tile that depends only on the tile (cached).

        Each color mode is a per-pixel tile term plus a per-cell color term, so the tile
        term is computed once here and the cell term is added by `MosaicBuilder`:

            Modulate  : tile*(1-s) + (tile_luma + cell - cell_luma)*s
                      = [tile*(1-s) + tile_luma*s]  +  [(cell - cell_luma)*s]
            Blend     : [tile*(1-s)]                +  [cell*s]
            Tile only : [tile]                      +  nothing

        A strength of 0 also means "raw tile". The tile term is rounded to uint8 here,
        once, so the full-size work in the builder stays in uint8 (4x less memory than
        float32). The price is rounding: a pixel can be 1 level off the float result.

        Parameters
        ----------
        tile_px : int, optional
            Tile resolution. Defaults to the manager's `tile_px`.
        color_mode : str, default "Modulate"
            One of `config.COLOR_MODES`.
        strength : float, default 0.85
            Recolor strength in [0, 1].

        Returns
        -------
        np.ndarray
            (N, T, T, 3) uint8, read-only.

        Example
        -------
        >>> TileManager("Halftone Dots").render_base(8, "Modulate", 0.85).shape
        (14, 8, 8, 3)
        """
        size = self._size(tile_px)
        color_mode = check_choice("color_mode", color_mode, config.COLOR_MODES)
        strength = float(strength)
        return self._cached(("base", size, color_mode, strength),
                            partial(self._compute_base, size, color_mode, strength))

    def contact_sheet(self, tile_px: int | None = None, cols: int = 8, cell_px: int = 48) -> np.ndarray:
        """Render all tiles as one preview image, for the web app's Tile sets tab.

        Parameters
        ----------
        tile_px : int, optional
            Resolution of the tiles to show. Defaults to the manager's `tile_px`.
        cols : int, default 8
            Tiles per row.
        cell_px : int, default 48
            Size each tile is enlarged to (nearest-neighbor, so its real pixels stay visible).

        Returns
        -------
        np.ndarray
            (H, W, 3) uint8 image on a dark background.

        Example
        -------
        >>> TileManager("LEGO Studs").contact_sheet(cols=8, cell_px=48).shape
        (114, 438, 3)
        """
        tiles = self.tiles(tile_px).tiles
        n, pad = len(tiles), 6
        rows = int(np.ceil(n / cols))
        sheet = np.full((rows * (cell_px + pad) + pad, cols * (cell_px + pad) + pad, 3),
                        24, dtype=np.uint8)
        for i, tile in enumerate(tiles):    # preview only: one paste per tile, at most ~50
            r, c = divmod(i, cols)
            y, x = pad + r * (cell_px + pad), pad + c * (cell_px + pad)
            sheet[y:y + cell_px, x:x + cell_px] = cv2.resize(
                tile, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
        return sheet


@lru_cache(maxsize=config.TILE_CACHE_SIZE)
def get_tile_manager(tile_set: str = config.DEFAULT_TILE_SET,
                     tile_px: int = config.DEFAULT_TILE_PX) -> TileManager:
    """Shared `TileManager` for a built-in set, so its caches survive between calls.

    The web app calls this instead of creating a new manager per request, which would
    throw the cached tile features away every time.

    Parameters
    ----------
    tile_set : str, default config.DEFAULT_TILE_SET
        Name of a built-in set.
    tile_px : int, default config.DEFAULT_TILE_PX
        Default tile resolution.

    Returns
    -------
    TileManager
        The same object for the same arguments.

    Example
    -------
    >>> get_tile_manager("Halftone Dots", 16) is get_tile_manager("Halftone Dots", 16)
    True
    """
    return TileManager(tile_set, tile_px)
