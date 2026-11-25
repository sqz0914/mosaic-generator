"""Configuration constants for the mosaic generator.

Every tunable number lives here, so the rest of the package has no hard-coded values.
The groups are:

* defaults     -- what `MosaicBuilder` and the web app use when nothing else is given
* choices      -- the allowed values of the string options
* limits       -- the ranges that input validation enforces
* internals    -- fixed parameters of the algorithm and its caches

Example:
    >>> from mosaic_generator import config
    >>> config.DEFAULT_GRID_SIZE
    48
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------

DEFAULT_GRID_SIZE = 48              # cells across the image
DEFAULT_TILE_SET = "Halftone Dots"
DEFAULT_TILE_PX = 16                # tile resolution in the output mosaic, in pixels
DEFAULT_COLOR_MODE = "Modulate"
DEFAULT_STRENGTH = 0.85             # how strongly the cell color tints its tile, 0..1
DEFAULT_MATCH_SPACE = "RGB"
DEFAULT_DETAIL_WEIGHT = 0.0         # 0 = match on tone only, 1 = match on texture only
DEFAULT_QUANTIZE_K = 0              # 0 or 1 = no color quantization
DEFAULT_PALETTE_K = 12              # palette size the web app offers when quantization is on
DEFAULT_LONG_EDGE = 768             # working resolution: the longest image side, in pixels

# --------------------------------------------------------------------------------------
# Choices
# --------------------------------------------------------------------------------------

COLOR_MODES = ("Tile only", "Blend", "Modulate")
MATCH_SPACES = ("RGB", "CIE-Lab")

# --------------------------------------------------------------------------------------
# Limits (checked by input validation)
# --------------------------------------------------------------------------------------

MIN_GRID_SIZE = 2
MAX_GRID_SIZE = 256

MIN_TILE_PX = 4
MAX_TILE_PX = 64

MIN_LONG_EDGE = 32
MAX_LONG_EDGE = 4096

MAX_QUANTIZE_K = 64

# Cells larger than this would make the float32 cell sums inexact (255 * 256**2 < 2**24).
MAX_CELL_PX = 256

# The mosaic is capped at this many pixels per side; finer grids get smaller tiles.
# Beyond this size the browser, not the algorithm, becomes the bottleneck.
MAX_OUTPUT_PX = 2048

# --------------------------------------------------------------------------------------
# Web app slider ranges
# --------------------------------------------------------------------------------------

UI_GRID_RANGE = (8, 128, 4)         # (min, max, step)
UI_TILE_PX_RANGE = (8, 32, 4)
UI_PALETTE_RANGE = (2, 32, 1)

# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

# Perceptual weights (ITU-R BT.601) used whenever a color collapses to one brightness.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMA.setflags(write=False)

# Tiles are drawn at this multiple of their final size (but at least TILE_RENDER_MIN_PX)
# and then area-averaged down, which gives much smoother edges than drawing small.
TILE_RENDER_SCALE = 4
TILE_RENDER_MIN_PX = 64

# How many (tile set, tile size) combinations, and how many derived tile features,
# are kept in memory.
TILE_CACHE_SIZE = 64

# How many reusable full-size work buffers each thread keeps (see utils.scratch).
MAX_SCRATCH_BUFFERS = 4

# Image file types accepted when loading an image or a folder of tiles.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
