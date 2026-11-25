"""Helper functions shared by the other modules.

* Input validation -- `check_int`, `check_float`, `check_choice` raise `MosaicInputError`
  with a message that names the parameter, the bad value and what is allowed.
* Files           -- `read_image_file` reads an image file as RGB, with clear errors.
* Timing          -- `StageTimer` measures each step of the pipeline, for the
                     processing-time display in the web app.
* Memory          -- `scratch` hands out reusable work arrays; `freeze` makes a cached
                     array read-only.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from . import config


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


class MosaicInputError(ValueError):
    """Raised when an argument or input image cannot be used to build a mosaic.

    It is a `ValueError`, so callers that already catch `ValueError` keep working.
    """


def check_int(name: str, value, low: int, high: int) -> int:
    """Check that `value` is a whole number in [low, high] and return it as an int.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : Any
        The value to check. Floats with no fractional part (e.g. 32.0 from a UI slider)
        are accepted.
    low, high : int
        Inclusive bounds.

    Returns
    -------
    int
        `value` converted to int.

    Raises
    ------
    MosaicInputError
        If `value` is not a whole number or lies outside [low, high].

    Example
    -------
    >>> check_int("grid_size", 32.0, 2, 256)
    32
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise MosaicInputError(f"{name} must be a whole number, got {value!r}") from None
    if not as_float.is_integer():
        raise MosaicInputError(f"{name} must be a whole number, got {value!r}")
    as_int = int(as_float)
    if not low <= as_int <= high:
        raise MosaicInputError(f"{name} must be between {low} and {high}, got {as_int}")
    return as_int


def check_float(name: str, value, low: float, high: float) -> float:
    """Check that `value` is a finite number in [low, high] and return it as a float.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : Any
        The value to check.
    low, high : float
        Inclusive bounds.

    Returns
    -------
    float
        `value` converted to float.

    Raises
    ------
    MosaicInputError
        If `value` is not a finite number or lies outside [low, high].

    Example
    -------
    >>> check_float("strength", 0.85, 0.0, 1.0)
    0.85
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise MosaicInputError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(as_float) or not low <= as_float <= high:
        raise MosaicInputError(f"{name} must be between {low} and {high}, got {value!r}")
    return as_float


def check_choice(name: str, value: str, choices: Sequence[str]) -> str:
    """Check that `value` is one of `choices` and return it.

    Parameters
    ----------
    name : str
        Parameter name, used in the error message.
    value : str
        The value to check.
    choices : sequence of str
        The allowed values.

    Returns
    -------
    str
        `value` unchanged.

    Raises
    ------
    MosaicInputError
        If `value` is not in `choices`.

    Example
    -------
    >>> check_choice("color_mode", "Blend", ("Tile only", "Blend", "Modulate"))
    'Blend'
    """
    if value not in choices:
        allowed = ", ".join(repr(c) for c in choices)
        raise MosaicInputError(f"{name} must be one of {allowed}, got {value!r}")
    return value


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------


def read_image_file(path: str | Path) -> np.ndarray:
    """Read an image file and return it as an RGB uint8 array.

    The file is read with NumPy and decoded with OpenCV, instead of `cv2.imread`,
    because `cv2.imread` fails on Windows paths that contain non-ASCII characters.
    Grayscale and RGBA files are converted to RGB.

    Parameters
    ----------
    path : str or Path
        Path to a PNG, JPEG, BMP, WebP or TIFF file.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB image.

    Raises
    ------
    MosaicInputError
        If the file does not exist, has an unsupported extension, or cannot be decoded.

    Example
    -------
    >>> img = read_image_file("assets/vibrant-sunflower-bloom-stockcake-256x256.jpg")  # doctest: +SKIP
    >>> img.shape                                                                       # doctest: +SKIP
    (256, 256, 3)
    """
    path = Path(path)
    if not path.is_file():
        raise MosaicInputError(f"image file not found: {path}")
    if path.suffix.lower() not in config.IMAGE_EXTENSIONS:
        allowed = ", ".join(config.IMAGE_EXTENSIONS)
        raise MosaicInputError(f"unsupported image type {path.suffix!r} for {path.name}; use {allowed}")

    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise MosaicInputError(f"cannot read {path.name}: the file is empty or not a valid image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------


class StageTimer:
    """Measure how long each named step of a computation takes.

    Use one `stage(...)` block per step; repeated names add up. Times are wall-clock
    milliseconds from `time.perf_counter`.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer.stage("match"):
    ...     pass
    >>> list(timer.stages_ms)
    ['match']
    """

    def __init__(self) -> None:
        """Start with no recorded stages.

        Returns
        -------
        None
        """
        self.stages_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the code inside the `with` block and record it under `name`.

        Parameters
        ----------
        name : str
            Label for this step, e.g. "Cell colors".

        Returns
        -------
        contextmanager
            Use as `with timer.stage(name): ...`. The elapsed milliseconds are added to
            `stages_ms[name]` when the block ends, even if it raises.

        Example
        -------
        >>> timer = StageTimer()
        >>> with timer.stage("resize"):
        ...     pass
        >>> "resize" in timer.stages_ms
        True
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.stages_ms[name] = self.stages_ms.get(name, 0.0) + elapsed

    @property
    def total_ms(self) -> float:
        """Total time of all recorded stages.

        Returns
        -------
        float
            Sum of `stages_ms`, in milliseconds.

        Example
        -------
        >>> StageTimer().total_ms
        0.0
        """
        return float(sum(self.stages_ms.values()))


# --------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------


def freeze(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous, read-only version of `arr`, for arrays kept in a cache.

    Cached arrays are shared between calls, so an accidental in-place write would
    silently change every later result. A read-only array raises an error instead.

    Parameters
    ----------
    arr : np.ndarray
        The array to protect.

    Returns
    -------
    np.ndarray
        A contiguous array with `writeable=False` (a copy only if `arr` was not
        contiguous).

    Example
    -------
    >>> freeze(np.zeros(3)).flags.writeable
    False
    """
    out = np.ascontiguousarray(arr)
    out.setflags(write=False)
    return out


_local = threading.local()


def scratch(shape: tuple[int, ...], dtype=np.float32, tag: str = "work") -> np.ndarray:
    """Return a reusable, uninitialized work array, kept between calls in this thread.

    Profiling showed that writing into newly allocated memory is slow for full-size
    images (a fresh 3 MB array made one copy about 4 times slower), so intermediate
    full-size arrays are reused instead of allocated on every call. Two rules keep this
    safe:

    * Only intermediate results go into these arrays. Callers always return a new
      array, so nobody holds a reference that a later call overwrites.
    * The arrays are per thread. The web app can serve several requests at the same
      time in different threads, and a shared array would let them overwrite each other.

    Each thread keeps at most `config.MAX_SCRATCH_BUFFERS` arrays and drops the least
    recently used one, so memory stays bounded.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    dtype : numpy dtype, default float32
        Element type.
    tag : str, default "work"
        Role of the array. Two arrays with the same shape and dtype but different tags
        are different arrays.

    Returns
    -------
    np.ndarray
        An array of `shape` and `dtype` whose contents are left over from earlier use;
        the caller must overwrite every element before reading it.

    Example
    -------
    >>> scratch((4, 4, 3), np.uint8, "offset") is scratch((4, 4, 3), np.uint8, "offset")
    True
    """
    cache: OrderedDict | None = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = OrderedDict()

    key = (tag, tuple(int(n) for n in shape), np.dtype(dtype).str)
    buf = cache.get(key)
    if buf is None:
        buf = np.empty(key[1], dtype=dtype)
        cache[key] = buf
        while len(cache) > config.MAX_SCRATCH_BUFFERS:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return buf
