"""Gradio web interface for the image mosaic generator.

Run locally:       python app.py
With hot reload:   gradio app.py

Three tabs:
* Mosaic       -- upload an image, choose the settings, see the mosaic, its similarity
                  scores and how long each processing step took.
* Tile sets    -- preview any built-in tile set at any resolution.
* Performance  -- processing time across grid sizes, with the current settings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from mosaic_generator import (
    COLOR_MODES,
    MATCH_SPACES,
    TILE_SETS,
    MosaicBuilder,
    MosaicInputError,
    MosaicResult,
    config,
    get_tile_manager,
    score,
)

IDLE_MESSAGE = "_Upload an image or pick an example, then press **Generate mosaic**._"
PERF_IDLE_MESSAGE = "_Press **Generate mosaic** on the Mosaic tab to measure the processing time._"
PERF_GRID_SIZES = (16, 32, 64, 128)
PERF_REPEATS = 5


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_builder(grid_size: float, tile_set_name: str, tile_px: float, color_mode: str,
                 strength: float, match_space: str, detail_weight: float,
                 use_quantization: bool, quantize_k: float) -> MosaicBuilder:
    """Create a MosaicBuilder from the values of the web controls.

    Parameters
    ----------
    grid_size : float
        Cells across (sliders give floats).
    tile_set_name : str
        Name of a built-in tile set.
    tile_px : float
        Tile resolution in pixels.
    color_mode : str
        One of `config.COLOR_MODES`.
    strength : float
        Recolor strength in [0, 1].
    match_space : str
        "RGB" or "CIE-Lab".
    detail_weight : float
        Texture weight in [0, 1].
    use_quantization : bool
        Whether to quantize the cell colors.
    quantize_k : float
        Palette size, used only when `use_quantization` is True.

    Returns
    -------
    MosaicBuilder
        A builder with a shared (cached) TileManager.

    Raises
    ------
    MosaicInputError
        If any value is out of range.
    """
    return MosaicBuilder(
        get_tile_manager(tile_set_name, int(tile_px)),
        grid_size=int(grid_size),
        color_mode=color_mode,
        strength=float(strength),
        match_space=match_space,
        detail_weight=float(detail_weight),
        quantize_k=int(quantize_k) if use_quantization else 0,
    )


def timing_table(stages_ms: dict[str, float]) -> str:
    """Format the processing time of each step as a Markdown table.

    Parameters
    ----------
    stages_ms : dict of str to float
        Milliseconds per step, in pipeline order.

    Returns
    -------
    str
        Markdown table with one row per step and a total row.

    Example
    -------
    >>> print(timing_table({"Preprocess": 0.1, "Reconstruction": 0.3}).splitlines()[-1])
    | **Total** | **0.40** | **100%** |
    """
    total = sum(stages_ms.values()) or 1e-9
    rows = ["| Step | Time (ms) | Share |", "|---|---:|---:|"]
    rows += [f"| {name} | {ms:.2f} | {ms / total:.0%} |" for name, ms in stages_ms.items()]
    rows.append(f"| **Total** | **{sum(stages_ms.values()):.2f}** | **100%** |")
    return "\n".join(rows)


def example_images() -> list[str]:
    """Write a few sample photos to a temporary folder for the Examples gallery.

    The photos come with scikit-image (`skimage.data`), so the repository needs no image
    files -- Hugging Face Spaces rejects binary files that are not stored with Xet/LFS.

    Returns
    -------
    list of str
        Paths to PNG files; empty if scikit-image's sample data cannot be loaded.
    """
    try:
        from skimage import data
        samples = {"astronaut": data.astronaut(), "coffee": data.coffee(), "cat": data.chelsea()}
    except Exception:                       # sample data missing: the app still works
        return []

    folder = Path(tempfile.gettempdir()) / "mosaic_generator_examples"
    folder.mkdir(exist_ok=True)
    paths = []
    for name, img in samples.items():
        path = folder / f"{name}.png"
        if not path.exists():
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    return paths


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------


def generate(image: np.ndarray | None, grid_size: float, tile_set_name: str, tile_px: float,
             color_mode: str, strength: float, match_space: str, detail_weight: float,
             use_quantization: bool, quantize_k: float):
    """Build the mosaic and everything shown beside it (Mosaic tab).

    Parameters
    ----------
    image : np.ndarray or None
        The uploaded image, or None if nothing is uploaded yet.
    grid_size, tile_set_name, tile_px, color_mode, strength, match_space, detail_weight,
    use_quantization, quantize_k
        Values of the web controls (see `make_builder`).

    Returns
    -------
    tuple
        (original, segmented, mosaic, summary markdown, timing markdown,
        results-panel visibility update).

    Raises
    ------
    gr.Error
        With the validation message, if the image or a setting cannot be used.
    """
    if image is None:                       # a prompt, not an error pop-up
        return None, None, None, IDLE_MESSAGE, "", gr.update(visible=False)

    try:
        builder = make_builder(grid_size, tile_set_name, tile_px, color_mode, strength,
                               match_space, detail_weight, use_quantization, quantize_k)
        result: MosaicResult = builder.build(image)
    except MosaicInputError as err:
        raise gr.Error(str(err)) from err

    s = score(result.original, result.mosaic)
    summary = (
        f"**Processing time: {result.elapsed_ms:.1f} ms** &nbsp;·&nbsp; "
        f"{result.n_cols} × {result.n_rows} grid ({result.n_rows * result.n_cols:,} cells) "
        f"&nbsp;·&nbsp; {result.cell_px} px cells → {result.tile_px} px tiles &nbsp;·&nbsp; "
        f"output {result.mosaic_full.shape[1]} × {result.mosaic_full.shape[0]} px\n\n"
        f"Similarity to the original: **MSE {s.mse:,.0f}** (lower is better) &nbsp;·&nbsp; "
        f"**SSIM {s.ssim:.4f}** (higher is better; 1.0 is identical)"
    )
    return (result.original, result.segmented, result.mosaic_full, summary,
            timing_table(result.stages_ms), gr.update(visible=True))


def measure_processing_time(image: np.ndarray | None, tile_set_name: str, tile_px: float,
                            color_mode: str, strength: float, match_space: str,
                            detail_weight: float, use_quantization: bool, quantize_k: float):
    """Measure the processing time across grid sizes (Performance tab).

    Each grid size is built once to warm up, then `PERF_REPEATS` times; the table shows
    the median time of each step.

    Parameters
    ----------
    image : np.ndarray or None
        The uploaded image, or None if nothing is uploaded yet.
    tile_set_name, tile_px, color_mode, strength, match_space, detail_weight,
    use_quantization, quantize_k
        Values of the Mosaic tab's controls (see `make_builder`).

    Returns
    -------
    tuple
        (table rows, status markdown, results-panel visibility update).
    """
    if image is None:
        return [], PERF_IDLE_MESSAGE, gr.update(visible=False)

    rows = []
    for grid in PERF_GRID_SIZES:            # one row per grid size in the table (4 sizes)
        try:
            builder = make_builder(grid, tile_set_name, tile_px, color_mode, strength,
                                   match_space, detail_weight, use_quantization, quantize_k)
            builder.build(image)            # warm-up
            runs = [builder.build(image) for _ in range(PERF_REPEATS)]
        except MosaicInputError as err:     # e.g. the image is too small for this grid
            rows.append([f"{grid}", "—", f"skipped: {err}", "", "", "", ""])
            continue
        stages = {name: float(np.median([r.stages_ms[name] for r in runs]))
                  for name in runs[0].stages_ms}
        rows.append([f"{grid}", f"{runs[0].n_rows * runs[0].n_cols:,}",
                     f"{sum(stages.values()):.2f}"] + [f"{ms:.2f}" for ms in stages.values()])

    status = (f"Median of {PERF_REPEATS} runs on a {image.shape[1]} × {image.shape[0]} image "
              f"at a {config.DEFAULT_LONG_EDGE} px working resolution, {tile_set_name} tiles "
              f"at {int(tile_px)} px.")
    return rows, status, gr.update(visible=True)


def preview_tiles(tile_set_name: str, tile_px: float):
    """Render a tile set's contact sheet (Tile sets tab).

    Parameters
    ----------
    tile_set_name : str
        Name of a built-in tile set.
    tile_px : float
        Tile resolution in pixels.

    Returns
    -------
    tuple of (np.ndarray, str)
        The preview image and a caption.
    """
    tm = get_tile_manager(tile_set_name, int(tile_px))
    caption = f"**{tm.name}** — {tm.n_tiles} tiles at {tm.tile_px} px. {tm.description}"
    return tm.contact_sheet(), caption


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------

with gr.Blocks(title="Image Mosaic Generator") as demo:
    gr.Markdown(
        "# Image Mosaic Generator\n"
        "Divide an image into a grid, match every cell to a tile, and rebuild the image out "
        "of tiles — in about a millisecond."
    )

    with gr.Tab("Mosaic"):
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Input image", type="numpy", height=260,
                                    sources=["upload", "clipboard"])
                examples = example_images()
                if examples:
                    gr.Examples(examples, inputs=image_in, label="Example images")

                g_min, g_max, g_step = config.UI_GRID_RANGE
                t_min, t_max, t_step = config.UI_TILE_PX_RANGE
                k_min, k_max, k_step = config.UI_PALETTE_RANGE
                grid_size = gr.Slider(g_min, g_max, value=config.DEFAULT_GRID_SIZE, step=g_step,
                                      label="Grid size (cells across)")
                tile_set_name = gr.Dropdown(TILE_SETS, value=config.DEFAULT_TILE_SET, label="Tile set")
                tile_px = gr.Slider(t_min, t_max, value=config.DEFAULT_TILE_PX, step=t_step,
                                    label="Tile resolution (px)")

                with gr.Accordion("Color", open=True):
                    color_mode = gr.Radio(list(COLOR_MODES), value=config.DEFAULT_COLOR_MODE,
                                          label="Color mode")
                    strength = gr.Slider(0.0, 1.0, value=config.DEFAULT_STRENGTH, step=0.05,
                                         label="Color strength")

                with gr.Accordion("Advanced", open=False):
                    match_space = gr.Radio(list(MATCH_SPACES), value=config.DEFAULT_MATCH_SPACE,
                                           label="Matching color space")
                    detail_weight = gr.Slider(0.0, 1.0, value=config.DEFAULT_DETAIL_WEIGHT, step=0.05,
                                              label="Texture matching (0 = tone only)")
                    use_quantization = gr.Checkbox(False, label="Quantize colors first")
                    quantize_k = gr.Slider(k_min, k_max, value=config.DEFAULT_PALETTE_K, step=k_step,
                                           label="Palette size (k)")

                run_btn = gr.Button("Generate mosaic", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown(IDLE_MESSAGE)
                with gr.Column(visible=False) as results:
                    with gr.Row():
                        original_out = gr.Image(label="1 · Original (preprocessed)", height=240)
                        segmented_out = gr.Image(label="2 · Segmented grid", height=240)
                    mosaic_out = gr.Image(label="3 · Mosaic", height=460)
                    with gr.Accordion("Processing time by step", open=True):
                        timing_out = gr.Markdown()

    with gr.Tab("Tile sets"):
        gr.Markdown(
            "Every tile is generated in code, so any tile set can be rendered at any "
            "resolution. Tiles are ordered by brightness — that ramp is what the matcher "
            "searches."
        )
        with gr.Row():
            preview_name = gr.Dropdown(TILE_SETS, value="ASCII Art", label="Tile set")
            preview_px = gr.Slider(t_min, t_max, value=config.DEFAULT_TILE_PX, step=t_step,
                                   label="Tile resolution (px)")
        preview_caption = gr.Markdown()
        preview_out = gr.Image(label="Tiles, ordered by brightness", height=380)

    with gr.Tab("Performance"):
        gr.Markdown(
            "How long the mosaic takes to build at different grid sizes, using the Mosaic "
            "tab's image and settings. Runs whenever you generate a mosaic."
        )
        perf_status = gr.Markdown(PERF_IDLE_MESSAGE)
        with gr.Column(visible=False) as perf_results:
            perf_table = gr.Dataframe(
                headers=["Grid", "Cells", "Total (ms)", "Preprocess (ms)", "Cell colors (ms)",
                         "Tile matching (ms)", "Reconstruction (ms)"],
                interactive=False, label="Processing time (median)",
            )

    # ---------------------------------------------------------------- wiring

    settings = [tile_set_name, tile_px, color_mode, strength, match_space, detail_weight,
                use_quantization, quantize_k]
    mosaic_outputs = [original_out, segmented_out, mosaic_out, summary, timing_out, results]

    # The timing sweep runs after the mosaic, so the mosaic appears first.
    run_btn.click(generate, [image_in, grid_size] + settings, mosaic_outputs).then(
        measure_processing_time, [image_in] + settings, [perf_table, perf_status, perf_results]
    )

    preview_name.change(preview_tiles, [preview_name, preview_px], [preview_out, preview_caption])
    preview_px.release(preview_tiles, [preview_name, preview_px], [preview_out, preview_caption])
    demo.load(preview_tiles, [preview_name, preview_px], [preview_out, preview_caption])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
