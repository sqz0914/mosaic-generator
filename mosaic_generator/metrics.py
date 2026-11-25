"""Similarity metrics between an image and its mosaic.

* MSE  -- mean squared error over all pixels and color channels (0 = identical).
* SSIM -- structural similarity from scikit-image (1 = identical).

Example:
    >>> img = np.zeros((16, 16, 3), np.uint8)
    >>> score(img, img)
    Scores(mse=0.0, ssim=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity

from .utils import MosaicInputError


@dataclass(frozen=True)
class Scores:
    """Both similarity metrics for one image pair.

    Attributes
    ----------
    mse : float
        Mean squared error, in squared gray levels (0 = identical).
    ssim : float
        Structural similarity in [-1, 1] (1 = identical); NaN for images smaller than
        3x3 pixels, where SSIM is not defined.
    """

    mse: float
    ssim: float


def _check_pair(reference: np.ndarray, candidate: np.ndarray) -> None:
    """Check that two images can be compared.

    Parameters
    ----------
    reference, candidate : np.ndarray
        (H, W, 3) images.

    Returns
    -------
    None

    Raises
    ------
    MosaicInputError
        If either is not an (H, W, 3) array or their shapes differ.
    """
    for name, img in (("reference", reference), ("candidate", candidate)):
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise MosaicInputError(f"{name} must be an (H, W, 3) image array, "
                                   f"got {getattr(img, 'shape', type(img).__name__)}")
    if reference.shape != candidate.shape:
        raise MosaicInputError(
            f"cannot compare images of different sizes: reference {reference.shape} "
            f"vs candidate {candidate.shape}")


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean squared error over all pixels and color channels.

    Parameters
    ----------
    reference, candidate : np.ndarray
        (H, W, 3) images of the same shape, values in [0, 255].

    Returns
    -------
    float
        Mean of the squared differences (0 = identical).

    Raises
    ------
    MosaicInputError
        If the images cannot be compared.

    Example
    -------
    >>> mse(np.zeros((2, 2, 3)), np.full((2, 2, 3), 2.0))
    4.0
    """
    _check_pair(reference, candidate)
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    return float(np.mean(diff ** 2))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Structural similarity over the three color channels.

    The window is 7x7, or smaller (odd, at least 3) for very small images.

    Parameters
    ----------
    reference, candidate : np.ndarray
        (H, W, 3) images of the same shape, values in [0, 255].

    Returns
    -------
    float
        SSIM in [-1, 1] (1 = identical), or NaN if the images are smaller than 3x3.

    Raises
    ------
    MosaicInputError
        If the images cannot be compared.

    Example
    -------
    >>> img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    >>> ssim(img, img)
    1.0
    """
    _check_pair(reference, candidate)
    smallest = min(reference.shape[0], reference.shape[1])
    win = min(7, smallest if smallest % 2 == 1 else smallest - 1)
    if win < 3:
        return float("nan")
    return float(structural_similarity(reference, candidate, channel_axis=2,
                                       data_range=255, win_size=win))


def score(reference: np.ndarray, candidate: np.ndarray) -> Scores:
    """Both metrics for one candidate image against a reference.

    Parameters
    ----------
    reference, candidate : np.ndarray
        (H, W, 3) images of the same shape, values in [0, 255].

    Returns
    -------
    Scores
        MSE and SSIM.

    Raises
    ------
    MosaicInputError
        If the images cannot be compared.

    Example
    -------
    >>> score(np.zeros((8, 8, 3)), np.zeros((8, 8, 3))).mse
    0.0
    """
    return Scores(mse=mse(reference, candidate), ssim=ssim(reference, candidate))
