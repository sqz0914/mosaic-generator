"""Unit tests for the mosaic_generator package.

Run from the project folder with `python -m pytest`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import threading

import cv2
import numpy as np
import pytest

import mosaic_generator
from baseline import core as lab1_core
from baseline import preprocess as lab1_preprocess
from baseline import tiles as lab1_tiles
from mosaic_generator import MosaicBuilder, build_mosaic, config
from mosaic_generator.metrics import mse, score, ssim
from mosaic_generator.mosaic_builder import gather_tiles, reconstruct, split_offset
from mosaic_generator.image_processor import (
    as_rgb_uint8,
    cell_statistics,
    crop_to_grid,
    grid_shape,
    load_image,
    prepare_image,
    quantize_colors,
    segmented_view,
    to_working_resolution,
)
from mosaic_generator.tile_manager import (
    TILE_SETS,
    TileManager,
    get_tile_manager,
    load_tile_directory,
)
from mosaic_generator.utils import (
    MosaicInputError,
    StageTimer,
    check_choice,
    check_float,
    check_int,
    freeze,
    read_image_file,
    scratch,
)


def write_image(path, img) -> None:
    """Write an RGB uint8 image to `path` (PNG or JPEG by extension).

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    img : np.ndarray
        (H, W, 3) RGB uint8 image.

    Returns
    -------
    None
    """
    ok, data = cv2.imencode(path.suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    data.tofile(str(path))


# ---------------------------------------------------------------- documentation


def _package_functions():
    """Every function and method defined in the mosaic_generator source files.

    Returns
    -------
    list of tuple (str, function)
        Qualified name and function object.
    """
    found = []
    for info in pkgutil.iter_modules(mosaic_generator.__path__):
        module = importlib.import_module(f"mosaic_generator.{info.name}")
        for _, obj in inspect.getmembers(module):
            members = [obj]
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                members = [m.fget if isinstance(m, property) else m for m in vars(obj).values()]
                members = [getattr(m, "__func__", m) for m in members]
            for fn in members:
                fn = inspect.unwrap(fn) if callable(fn) else fn
                if (inspect.isfunction(fn) and fn.__module__ == module.__name__
                        and fn.__code__.co_filename.endswith(".py")):
                    found.append((f"{module.__name__}.{fn.__qualname__}", fn))
    return found


@pytest.mark.parametrize("name, fn", _package_functions(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_function_has_a_full_docstring(name, fn):
    """Lab requirement: a description, parameters with types, and the return value."""
    doc = inspect.getdoc(fn) or ""
    assert doc.strip(), f"{name} has no docstring"
    params = [p for p in inspect.signature(fn).parameters if p not in ("self", "cls")]
    if params:
        assert "Parameters" in doc, f"{name} does not document its parameters"
    assert "Returns" in doc, f"{name} does not document its return value"


# ---------------------------------------------------------------- utils: validation


@pytest.mark.parametrize("value, expected", [(32, 32), (32.0, 32), ("16", 16), (2, 2), (256, 256)])
def test_check_int_accepts_whole_numbers_in_range(value, expected):
    assert check_int("grid_size", value, 2, 256) == expected


@pytest.mark.parametrize("value", [1, 257, 3.5, "abc", None])
def test_check_int_rejects_bad_values_with_the_parameter_name(value):
    with pytest.raises(MosaicInputError, match="grid_size"):
        check_int("grid_size", value, 2, 256)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf"), "x"])
def test_check_float_rejects_bad_values(value):
    with pytest.raises(MosaicInputError, match="strength"):
        check_float("strength", value, 0.0, 1.0)


def test_check_choice_lists_the_allowed_values():
    assert check_choice("color_mode", "Blend", config.COLOR_MODES) == "Blend"
    with pytest.raises(MosaicInputError, match="'Tile only', 'Blend', 'Modulate'"):
        check_choice("color_mode", "Sepia", config.COLOR_MODES)


def test_input_error_is_a_value_error():
    """Callers that already catch ValueError keep working."""
    assert issubclass(MosaicInputError, ValueError)


# ---------------------------------------------------------------- utils: timing and memory


def test_stage_timer_records_and_sums_stages():
    timer = StageTimer()
    with timer.stage("a"):
        pass
    with timer.stage("b"):
        pass
    with timer.stage("a"):
        pass
    assert list(timer.stages_ms) == ["a", "b"]
    assert timer.total_ms == pytest.approx(sum(timer.stages_ms.values()))
    assert all(ms >= 0 for ms in timer.stages_ms.values())


def test_freeze_makes_arrays_read_only():
    arr = freeze(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        arr[0, 0] = 1


def test_scratch_reuses_per_tag_and_stays_bounded():
    a = scratch((8, 8, 3), np.uint8, "x")
    assert scratch((8, 8, 3), np.uint8, "x") is a
    assert scratch((8, 8, 3), np.uint8, "y") is not a
    for n in range(config.MAX_SCRATCH_BUFFERS + 3):
        scratch((n + 1, 2))
    from mosaic_generator.utils import _local
    assert len(_local.cache) <= config.MAX_SCRATCH_BUFFERS


def test_scratch_buffers_are_separate_per_thread():
    """Two requests running at the same time must never share a work buffer."""
    got, barrier = [], threading.Barrier(2)

    def worker():
        got.append(scratch((8, 8, 3)))
        barrier.wait()                      # keep both threads alive until both have one

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got[0] is not got[1]


# ---------------------------------------------------------------- utils: reading images


def test_read_image_file_returns_rgb(tmp_path):
    img = np.zeros((10, 20, 3), np.uint8)
    img[..., 0] = 200                                   # pure red in RGB
    path = tmp_path / "red.png"
    write_image(path, img)
    out = read_image_file(path)
    assert out.shape == (10, 20, 3)
    assert out[0, 0].tolist() == [200, 0, 0]


def test_read_image_file_handles_non_ascii_paths(tmp_path):
    path = tmp_path / "向日葵.png"
    write_image(path, np.full((4, 4, 3), 90, np.uint8))
    assert read_image_file(path).shape == (4, 4, 3)


def test_read_image_file_converts_grayscale_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    ok, data = cv2.imencode(".png", np.full((5, 5), 128, np.uint8))
    data.tofile(str(path))
    assert read_image_file(path).shape == (5, 5, 3)


@pytest.mark.parametrize("name, content, message", [
    ("missing.png", None, "not found"),
    ("notes.txt", b"hello", "unsupported image type"),
    ("broken.png", b"not really a png", "not a valid image"),
    ("empty.jpg", b"", "not a valid image"),
])
def test_read_image_file_gives_clear_errors(tmp_path, name, content, message):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(MosaicInputError, match=message):
        read_image_file(path)


# ---------------------------------------------------------------- tile_manager: built-in sets


@pytest.mark.parametrize("name", TILE_SETS)
@pytest.mark.parametrize("tile_px", [8, 16])
def test_built_in_sets_are_well_formed(name, tile_px):
    ts = TileManager(name, tile_px).tiles()
    assert ts.tiles.shape == (ts.n_tiles, tile_px, tile_px, 3)
    assert ts.tiles.dtype == np.uint8
    assert ts.keys.shape == (ts.n_tiles, 3)
    assert ts.mode in ("intensity", "color")
    assert np.all(np.diff(ts.luma_keys) >= 0), "keys must be sorted by brightness"


@pytest.mark.parametrize("name", TILE_SETS)
def test_built_in_sets_span_the_tonal_range(name):
    """A set that cannot reach dark or bright cells silently caps mosaic quality."""
    luma = TileManager(name).tiles().luma_keys
    assert luma.min() < 40, "no tile dark enough for shadow cells"
    assert luma.max() > 215, "no tile bright enough for highlight cells"
    assert np.diff(luma).max() < 55, "gap in the ramp leaves a tone unrepresentable"


@pytest.mark.parametrize("name", TILE_SETS)
@pytest.mark.parametrize("tile_px", [4, 12, 16])
def test_built_in_tiles_are_identical_to_lab1(name, tile_px):
    """Refactoring the drawing code must not change a single tile pixel."""
    new = TileManager(name).tiles(tile_px)
    old = lab1_tiles.get_tile_set(name, tile_px)
    np.testing.assert_array_equal(new.tiles, old.tiles)
    np.testing.assert_array_equal(new.keys, old.keys)
    assert new.mode == old.mode


def test_tile_features_have_the_right_shapes_and_are_cached():
    tm = TileManager("LEGO Studs", 8)
    assert tm.match_keys("RGB").shape == (16, 3)
    assert tm.match_keys("CIE-Lab").shape == (16, 3)
    assert tm.tile_detail().shape == (16,)
    assert tm.render_base(8, "Modulate", 0.85).shape == (16, 8, 8, 3)
    assert tm.render_base(8, "Modulate", 0.85) is tm.render_base(8, "Modulate", 0.85)
    assert TileManager("Halftone Dots").match_keys("RGB").shape == (14, 1)


def test_cached_tile_features_are_read_only():
    tm = TileManager("Halftone Dots", 16)
    for arr in (tm.tiles().tiles, tm.match_keys(), tm.tile_detail(), tm.render_base()):
        assert not arr.flags.writeable


def test_tile_only_base_is_the_raw_tiles():
    tm = TileManager("ASCII Art", 8)
    np.testing.assert_array_equal(tm.render_base(8, "Tile only", 0.85), tm.tiles(8).tiles)
    np.testing.assert_array_equal(tm.render_base(8, "Modulate", 0.0), tm.tiles(8).tiles)


def test_get_tile_manager_shares_one_manager():
    assert get_tile_manager("Halftone Dots", 16) is get_tile_manager("Halftone Dots", 16)
    assert get_tile_manager("Halftone Dots", 16) is not get_tile_manager("ASCII Art", 16)


def test_contact_sheet_shows_every_tile():
    sheet = TileManager("LEGO Studs").contact_sheet(cols=8, cell_px=48)
    assert sheet.shape == (2 * 54 + 6, 8 * 54 + 6, 3)


@pytest.mark.parametrize("kwargs, message", [
    (dict(tile_set="Mosaic Glass"), "tile_set must be one of"),
    (dict(tile_px=2), "tile_px must be between"),
    (dict(tile_px=500), "tile_px must be between"),
    (dict(tile_set="Halftone Dots", tile_directory="."), "not both"),
])
def test_tile_manager_rejects_bad_arguments(kwargs, message):
    with pytest.raises(MosaicInputError, match=message):
        TileManager(**kwargs)


def test_tile_manager_rejects_bad_feature_arguments():
    tm = TileManager()
    with pytest.raises(MosaicInputError, match="match_space"):
        tm.match_keys("HSV")
    with pytest.raises(MosaicInputError, match="color_mode"):
        tm.render_base(color_mode="Sepia")


# ---------------------------------------------------------------- tile_manager: tile folders


@pytest.fixture
def tile_folder(tmp_path):
    """A folder with three good tiles (one not square), one broken file and one text file."""
    for i, value in enumerate((30, 130, 230)):
        write_image(tmp_path / f"tile_{i}.png", np.full((20, 20 + 10 * (i == 2), 3), value, np.uint8))
    (tmp_path / "broken.jpg").write_bytes(b"garbage")
    (tmp_path / "readme.txt").write_text("not an image")
    return tmp_path


def test_tile_folder_loads_good_tiles_and_skips_broken_ones(tile_folder):
    with pytest.warns(UserWarning, match="broken.jpg"):
        tm = TileManager(tile_directory=tile_folder, tile_px=8)
    assert tm.n_tiles == 3
    assert tm.skipped_files == ["broken.jpg"]
    assert tm.mode == "color"
    assert tm.tiles().tiles.shape == (3, 8, 8, 3)
    assert tm.tiles(12).tiles.shape == (3, 12, 12, 3)


def test_tile_folder_crops_images_to_squares(tile_folder):
    with pytest.warns(UserWarning):
        images, _ = load_tile_directory(tile_folder)
    assert all(img.shape[0] == img.shape[1] for img in images)


def test_missing_tile_folder_is_a_clear_error(tmp_path):
    with pytest.raises(MosaicInputError, match="not found"):
        TileManager(tile_directory=tmp_path / "nope")


def test_tile_folder_without_images_is_a_clear_error(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    with pytest.raises(MosaicInputError, match="no tile images"):
        TileManager(tile_directory=tmp_path)


def test_tile_folder_with_too_few_readable_images_is_a_clear_error(tmp_path):
    write_image(tmp_path / "only.png", np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "broken.png").write_bytes(b"garbage")
    with pytest.warns(UserWarning), pytest.raises(MosaicInputError, match="at least 2 readable"):
        TileManager(tile_directory=tmp_path)


# ---------------------------------------------------------------- image_processor


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    """A deterministic 480x640 image with gradients, flat regions and hard edges.

    Returns
    -------
    np.ndarray
        (480, 640, 3) uint8 RGB image.
    """
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w * 255, yy / h * 255, (xx + yy) / (w + h) * 255], -1).astype(np.uint8)
    cv2.circle(img, (200, 200), 120, (250, 40, 40), -1)
    cv2.rectangle(img, (400, 250), (600, 430), (20, 200, 90), -1)
    return img


@pytest.mark.parametrize("image, shape", [
    (np.full((10, 12), 128, np.uint8), (10, 12, 3)),           # grayscale
    (np.full((10, 12, 1), 128, np.uint8), (10, 12, 3)),        # single channel
    (np.full((10, 12, 4), 200, np.uint8), (10, 12, 3)),        # RGBA
    (np.full((10, 12, 3), 0.5, np.float32), (10, 12, 3)),      # float in [0, 1]
    (np.full((10, 12, 3), 300.0), (10, 12, 3)),                # float above 255
])
def test_as_rgb_uint8_accepts_common_formats(image, shape):
    out = as_rgb_uint8(image)
    assert out.shape == shape and out.dtype == np.uint8 and out.flags.c_contiguous


@pytest.mark.parametrize("image, message", [
    (None, "no image"),
    (np.zeros((0, 10, 3), np.uint8), "empty"),
    (np.zeros((4, 4, 2), np.uint8), "shape"),
    (np.zeros((2, 4, 4, 3), np.uint8), "shape"),
    (np.full((4, 4, 3), np.nan), "NaN"),
    (np.array([["a", "b"]]), "numeric"),
])
def test_as_rgb_uint8_rejects_unusable_images(image, message):
    with pytest.raises(MosaicInputError, match=message):
        as_rgb_uint8(image)


def test_load_image_reads_files_and_arrays(tmp_path, sample):
    path = tmp_path / "sample.png"
    write_image(path, sample)
    np.testing.assert_array_equal(load_image(path), sample)
    np.testing.assert_array_equal(load_image(str(path)), sample)
    np.testing.assert_array_equal(load_image(sample), sample)


def test_working_resolution_matches_lab1(sample):
    for long_edge in (256, 640, 768, 1024):
        np.testing.assert_array_equal(to_working_resolution(sample, long_edge),
                                      lab1_preprocess.to_working_resolution(sample, long_edge))


@pytest.mark.parametrize("long_edge", [0, 31, 5000, 12.5])
def test_working_resolution_rejects_bad_long_edge(sample, long_edge):
    with pytest.raises(MosaicInputError, match="long_edge"):
        to_working_resolution(sample, long_edge)


@pytest.mark.parametrize("grid_size", [7, 16, 31, 32, 64, 97])
def test_crop_to_grid_matches_lab1_and_divides_exactly(sample, grid_size):
    grid = crop_to_grid(sample, grid_size)
    old, n_rows, n_cols, cell = lab1_preprocess.crop_to_grid(sample, grid_size)
    assert (grid.n_rows, grid.n_cols, grid.cell_px) == (n_rows, n_cols, cell)
    np.testing.assert_array_equal(grid.image, old)
    assert grid.image.shape[:2] == (grid.n_rows * grid.cell_px, grid.n_cols * grid.cell_px)


def test_crop_discards_less_than_one_cell(sample):
    work = to_working_resolution(sample, 768)
    grid = crop_to_grid(work, 37)
    assert work.shape[0] - grid.image.shape[0] < grid.cell_px
    assert work.shape[1] - grid.image.shape[1] < grid.cell_px


@pytest.mark.parametrize("height, width, grid_size, message", [
    (100, 100, 1, "grid_size must be between"),
    (100, 100, 300, "grid_size must be between"),
    (100, 20, 32, "20 px wide, too small"),
    (10, 640, 16, "shorter than one 40 px cell"),
    (600, 600, 2, "more than the 256 px limit"),
])
def test_grid_shape_explains_why_a_grid_does_not_fit(height, width, grid_size, message):
    with pytest.raises(MosaicInputError, match=message):
        grid_shape(height, width, grid_size)


def test_prepare_image_runs_load_resize_and_crop(sample):
    grid = prepare_image(sample, grid_size=32, long_edge=512)
    assert (grid.n_rows, grid.n_cols, grid.cell_px) == (24, 32, 16)


@pytest.mark.parametrize("n_rows, n_cols, cell", [(32, 32, 16), (6, 8, 7), (16, 16, 64), (3, 5, 1), (4, 4, 96)])
def test_cell_means_equal_lab1_bit_for_bit(n_rows, n_cols, cell):
    """The fast two-pass means must equal Lab 1's reshape-mean exactly."""
    img = np.random.default_rng(cell).integers(0, 256, (n_rows * cell, n_cols * cell, 3), dtype=np.uint8)
    new, _ = cell_statistics(img, n_rows, n_cols, cell)
    old, _ = lab1_core.cell_statistics(img, n_rows, n_cols, cell, with_std=False)
    assert new.dtype == np.float32
    np.testing.assert_array_equal(new, old)


def test_cell_statistics_with_std_matches_lab1(sample):
    grid = crop_to_grid(sample, 40)
    new_mean, new_std = cell_statistics(*grid, with_std=True)
    old_mean, old_std = lab1_core.cell_statistics(*grid, with_std=True)
    np.testing.assert_allclose(new_mean, old_mean, rtol=1e-6)
    np.testing.assert_allclose(new_std, old_std, rtol=1e-6)


def test_flat_cells_have_zero_contrast():
    mean, std = cell_statistics(np.full((32, 32, 3), 77, np.uint8), 4, 4, 8, with_std=True)
    assert np.allclose(mean, 77) and np.allclose(std, 0)


def test_quantize_reduces_the_palette(sample):
    out = quantize_colors(sample, k=8)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 8
    with pytest.raises(MosaicInputError, match="quantize_k"):
        quantize_colors(sample, k=500)


def test_segmented_view_fills_each_cell_with_its_color():
    cells = np.array([[[10, 20, 30], [40, 50, 60]]], np.float32)
    view = segmented_view(cells, 4, 8)
    assert view.shape == (4, 8, 3)
    assert view[0, 0].tolist() == [10, 20, 30] and view[3, 7].tolist() == [40, 50, 60]


# ---------------------------------------------------------------- metrics


def test_identical_images_score_perfectly(sample):
    s = score(sample, sample)
    assert s.mse == 0 and s.ssim == pytest.approx(1.0)


def test_metrics_match_lab1(sample):
    from baseline import metrics as lab1_metrics
    other = np.roll(sample, 7, axis=1)
    assert mse(sample, other) == lab1_metrics.mse(sample, other)
    assert ssim(sample, other) == lab1_metrics.ssim(sample, other)


def test_ssim_is_nan_for_tiny_images():
    assert np.isnan(ssim(np.zeros((2, 2, 3)), np.zeros((2, 2, 3))))


@pytest.mark.parametrize("a, b, message", [
    (np.zeros((4, 4, 3)), np.zeros((4, 5, 3)), "different sizes"),
    (np.zeros((4, 4)), np.zeros((4, 4)), r"\(H, W, 3\)"),
    ([1, 2, 3], np.zeros((4, 4, 3)), r"\(H, W, 3\)"),
])
def test_metrics_reject_images_that_cannot_be_compared(a, b, message):
    with pytest.raises(MosaicInputError, match=message):
        score(a, b)


# ---------------------------------------------------------------- mosaic_builder: building blocks


@pytest.mark.parametrize("tile_px", [4, 8, 33])
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (32, 32)])
def test_gather_tiles_places_every_tile(tile_px, shape):
    tiles = TileManager("ASCII Art").tiles(tile_px).tiles
    indices = np.random.default_rng(tile_px).integers(0, len(tiles), shape).astype(np.int32)
    ref = tiles[indices].transpose(0, 2, 1, 3, 4).reshape(shape[0] * tile_px, shape[1] * tile_px, 3)
    np.testing.assert_array_equal(gather_tiles(tiles, indices), ref)


def test_split_offset_rounds_and_separates_signs():
    pos, neg = split_offset(np.array([[[300.0, -12.6, 0.4]]]))
    assert pos.tolist() == [[[255, 0, 0]]] and neg.tolist() == [[[0, 13, 0]]]


@pytest.mark.parametrize("color_mode", config.COLOR_MODES)
@pytest.mark.parametrize("tile_px, shape", [(16, (32, 32)), (7, (3, 5)), (4, (1, 9))])
def test_reconstruct_equals_the_plain_formula(color_mode, tile_px, shape):
    """clip(rounded tile part + rounded cell part, 0, 255), placed cell by cell."""
    from mosaic_generator.mosaic_builder import cell_offset
    tm = TileManager("LEGO Studs", 8)
    rng = np.random.default_rng(tile_px)
    indices = rng.integers(0, tm.n_tiles, shape).astype(np.int32)
    cell_mean = rng.uniform(0, 255, shape + (3,)).astype(np.float32)

    ref = tm.render_base(tile_px, color_mode, 0.85)[indices].astype(np.int32)
    offset = cell_offset(cell_mean, color_mode, 0.85)
    if offset is not None:
        ref += np.rint(offset).astype(np.int32)[:, :, None, None, :]
    ref = np.clip(ref, 0, 255).astype(np.uint8).transpose(0, 2, 1, 3, 4)
    ref = ref.reshape(shape[0] * tile_px, shape[1] * tile_px, 3)
    np.testing.assert_array_equal(reconstruct(indices, tm, tile_px, cell_mean, color_mode, 0.85), ref)


# ---------------------------------------------------------------- mosaic_builder: same output as Lab 1


def assert_same_mosaic(new, ref, max_pixel_diff: int = 1) -> None:
    """Check that two mosaic results agree.

    Parameters
    ----------
    new : MosaicResult
        Result from mosaic_generator.
    ref : baseline.core.MosaicResult
        Result from the Lab 1 code.
    max_pixel_diff : int, default 1
        Largest allowed difference of one pixel value (uint8 rounding).

    Returns
    -------
    None
    """
    np.testing.assert_array_equal(new.indices, ref.indices)
    np.testing.assert_array_equal(new.original, ref.original)
    np.testing.assert_array_equal(new.segmented, ref.segmented)
    assert new.mosaic_full.shape == ref.mosaic_full.shape
    assert new.mosaic.shape == ref.mosaic.shape == ref.original.shape
    diff = np.abs(new.mosaic_full.astype(int) - ref.mosaic_full.astype(int))
    assert diff.max() <= max_pixel_diff


@pytest.mark.parametrize("size, grid", [(256, 16), (512, 32), (1024, 64)])
def test_matches_lab1_loop_on_the_required_configurations(sample, size, grid):
    img = cv2.resize(sample, (size, size), interpolation=cv2.INTER_AREA)
    kw = dict(grid_size=grid, tile_px=size // grid, long_edge=size)
    assert_same_mosaic(build_mosaic(img, **kw), lab1_core.build_mosaic_loop(img, **kw))


@pytest.mark.parametrize("tile_set", TILE_SETS)
@pytest.mark.parametrize("color_mode", config.COLOR_MODES)
def test_matches_lab1_loop_for_every_tile_set_and_color_mode(sample, tile_set, color_mode):
    kw = dict(grid_size=32, tile_set_name=tile_set, tile_px=12, color_mode=color_mode)
    assert_same_mosaic(build_mosaic(sample, **kw), lab1_core.build_mosaic_loop(sample, **kw))


@pytest.mark.parametrize("tile_set", TILE_SETS)
@pytest.mark.parametrize("space", config.MATCH_SPACES)
@pytest.mark.parametrize("detail_weight", [0.0, 0.5])
def test_matches_lab1_for_match_options(sample, tile_set, space, detail_weight):
    """The loop baseline only supports RGB without texture, so compare with Lab 1 vectorized."""
    kw = dict(grid_size=40, tile_set_name=tile_set, match_space=space, detail_weight=detail_weight)
    assert_same_mosaic(build_mosaic(sample, **kw), lab1_core.build_mosaic(sample, **kw))


@pytest.mark.parametrize("quantize_k", [4, 12])
def test_matches_lab1_with_quantization(sample, quantize_k):
    kw = dict(grid_size=48, tile_set_name="LEGO Studs", quantize_k=quantize_k)
    cv2.setRNGSeed(0)                     # k-means starts from random centers
    new = build_mosaic(sample, **kw)
    cv2.setRNGSeed(0)
    assert_same_mosaic(new, lab1_core.build_mosaic(sample, **kw))


@pytest.mark.parametrize("strength", [0.0, 0.3, 1.0])
def test_matches_lab1_across_strength(sample, strength):
    kw = dict(grid_size=24, strength=strength)
    assert_same_mosaic(build_mosaic(sample, **kw), lab1_core.build_mosaic(sample, **kw))


def test_large_grid_shrinks_tiles_like_lab1(sample):
    kw = dict(grid_size=128, tile_px=32)
    new, ref = build_mosaic(sample, **kw), lab1_core.build_mosaic(sample, **kw)
    assert new.tile_px == ref.tile_px < 32
    assert max(new.mosaic_full.shape[:2]) <= config.MAX_OUTPUT_PX
    assert_same_mosaic(new, ref)


@pytest.mark.parametrize("image", [
    np.full((100, 100), 128, np.uint8),             # grayscale
    np.full((100, 100, 4), 200, np.uint8),          # RGBA
    np.zeros((640, 120, 3), np.uint8),              # tall portrait
    np.zeros((120, 640, 3), np.uint8),              # wide landscape
])
def test_awkward_inputs_still_make_a_mosaic(image):
    result = build_mosaic(image, grid_size=24, tile_px=8)
    assert result.mosaic.shape == result.original.shape and result.mosaic.dtype == np.uint8


# ---------------------------------------------------------------- MosaicBuilder API


def test_builder_api_from_the_lab_handout(sample):
    builder = MosaicBuilder(TileManager("Halftone Dots", tile_px=16), grid_size=(32, 32))
    mosaic = builder.create_mosaic(sample)
    similarity = builder.compute_similarity(sample, mosaic)
    assert mosaic.ndim == 3 and mosaic.dtype == np.uint8
    assert similarity.mse > 0 and -1 <= similarity.ssim <= 1


def test_build_reports_the_time_of_each_step(sample):
    result = MosaicBuilder(TileManager(), grid_size=32).build(sample)
    assert list(result.stages_ms) == ["Preprocess", "Cell colors", "Tile matching", "Reconstruction"]
    assert result.elapsed_ms == pytest.approx(sum(result.stages_ms.values()))
    assert result.elapsed_ms > 0


def test_builder_accepts_a_file_path(tmp_path, sample):
    path = tmp_path / "input.jpg"
    write_image(path, sample)
    assert MosaicBuilder(TileManager(tile_px=8), grid_size=16).create_mosaic(path).ndim == 3


def test_builder_works_with_a_tile_folder(tmp_path, sample):
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200), (240, 240, 240)]):
        write_image(tmp_path / f"t{i}.png", np.full((10, 10, 3), color, np.uint8))
    builder = MosaicBuilder(TileManager(tile_directory=tmp_path, tile_px=8), grid_size=16)
    result = builder.build(sample)
    assert result.indices.max() < 4 and result.mosaic_full.shape[2] == 3


def test_finer_grids_lower_the_error(sample):
    """More cells must give a closer reconstruction."""
    errors = [score(r.original, r.mosaic).mse
              for r in (build_mosaic(sample, grid_size=g, tile_px=16) for g in (16, 32, 64, 128))]
    assert errors == sorted(errors, reverse=True), errors


def test_repeated_and_concurrent_builds_give_the_same_mosaics(sample):
    """Work buffers are reused and per thread; results must never share them."""
    from concurrent.futures import ThreadPoolExecutor
    first = build_mosaic(sample, grid_size=32)
    kept = first.mosaic_full.copy()
    build_mosaic(255 - sample, grid_size=32)
    np.testing.assert_array_equal(first.mosaic_full, kept)

    images = [sample, 255 - sample, np.ascontiguousarray(sample[:, ::-1])] * 4
    expected = [build_mosaic(img, grid_size=32).mosaic_full for img in images]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda img: build_mosaic(img, grid_size=32).mosaic_full, images))
    for e, g in zip(expected, got):
        np.testing.assert_array_equal(e, g)


@pytest.mark.parametrize("kwargs, message", [
    (dict(tile_manager="Halftone Dots"), "must be a TileManager"),
    (dict(grid_size=1), "grid_size must be between"),
    (dict(grid_size=(32, 32, 3)), "grid_size must be an int"),
    (dict(color_mode="Sepia"), "color_mode must be one of"),
    (dict(strength=1.5), "strength must be between"),
    (dict(match_space="HSV"), "match_space must be one of"),
    (dict(detail_weight=-0.1), "detail_weight must be between"),
    (dict(quantize_k=100), "quantize_k must be between"),
    (dict(long_edge=10_000), "long_edge must be between"),
])
def test_builder_rejects_bad_settings(kwargs, message):
    kwargs.setdefault("tile_manager", TileManager())
    with pytest.raises(MosaicInputError, match=message):
        MosaicBuilder(**kwargs)


@pytest.mark.parametrize("image, message", [
    (None, "no image"),
    (np.zeros((0, 0, 3), np.uint8), "empty"),
    ("does_not_exist.png", "not found"),
])
def test_builder_rejects_bad_images(image, message):
    with pytest.raises(MosaicInputError, match=message):
        MosaicBuilder(TileManager()).create_mosaic(image)


def test_build_mosaic_rejects_unknown_tile_sets(sample):
    with pytest.raises(MosaicInputError, match="tile_set_name"):
        build_mosaic(sample, tile_set_name="Mosaic Glass")
