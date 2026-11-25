"""Image mosaic generator: rebuild an image out of small tiles, fast.

Modules
-------
config           Configuration constants (defaults, allowed values, limits).
image_processor  Image loading, resizing, grid division and cell colors.
tile_manager     TileManager: built-in or folder tiles, caching, feature extraction.
mosaic_builder   MosaicBuilder: tile matching and mosaic reconstruction.
metrics          Similarity metrics (MSE, SSIM).
utils            Input validation, timing and memory helpers.

Example
-------
>>> import numpy as np
>>> from mosaic_generator import MosaicBuilder, TileManager
>>> tile_manager = TileManager("Halftone Dots", tile_px=16)
>>> builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
>>> image = np.zeros((512, 512, 3), np.uint8)
>>> mosaic = builder.create_mosaic(image)
>>> similarity = builder.compute_similarity(image, mosaic)
"""

from .config import COLOR_MODES, MATCH_SPACES
from .metrics import Scores, score
from .mosaic_builder import MosaicBuilder, MosaicResult, build_mosaic
from .tile_manager import TILE_SETS, TileManager, TileSet, get_tile_manager
from .utils import MosaicInputError

__all__ = [
    "MosaicBuilder",
    "TileManager",
    "MosaicResult",
    "TileSet",
    "Scores",
    "MosaicInputError",
    "build_mosaic",
    "get_tile_manager",
    "score",
    "TILE_SETS",
    "COLOR_MODES",
    "MATCH_SPACES",
]
