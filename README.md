# 🧬 MALDI Deconvolution RL

A comprehensive pipeline for processing and deconvolving MALDI imaging mass spectrometry data using Richardson-Lucy deconvolution with realistic PSF modeling.

## ⚙ Prerequisites

This project requires **[uv](https://docs.astral.sh/uv/#installation)**

<details>


**macOS & Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, see the [official uv docs](https://docs.astral.sh/uv/#installation).

</details>

## 📥 Clone the Repository

```bash
git clone https://github.com/MMV-Lab/maldi-deconvolution.git
cd maldi-deconvolution
```



## 🚀 Quick Start

This project is powered by **[marimo](https://marimo.io/)** (interactive Python notebooks that run as apps) and managed with **[uv](https://docs.astral.sh/uv/)** for dependency resolution.

**Run the notebooks (end users):**
```bash
uv run marimo run ./notebooks
```

**Open notebooks in edit mode (developers):**
```bash
uv run marimo edit
```

`uv` will automatically resolve and install all dependencies from `pyproject.toml` and `uv.lock` on first run.

## 🔬 Workflow

The pipeline is split into the following sequential notebooks:

### `1a_extract_data.py` — Ion Image Extraction
Load one or more `.imzML` files and extract ion images for selected m/z values (single or batch via CSV). Scans the full dataset to determine m/z ranges and pixel sizes, then exports a structured `.h5` (NetCDF) file ready for deconvolution or visualization.

### `1b_extract_decay_profile.py` — Signal Decay Analysis
Tracks signal intensity across **multiple repeated laser passes** on the same tissue section. Auto-detects the top spectral peaks, builds a `xarray` 4D data cube (pass × m/z × y × x).Exports a `Decay_Profile.csv` to be used in the deconvolution step.

### `2_deconvolve.py` — PSF Simulation & Deconvolution
The core processing step. Given instrument parameters (laser diameter, step size, decay profile), it simulates a realistic PSF and applies Richardson-Lucy deconvolution to the ion image stack. Outputs a multi-iteration deconvolved `.h5` file, each iteration slice can be inspected independently.

### `3_visualize.py` — Interactive Visualization & Export
Browse deconvolved or original `.h5` files interactively. Adjust m/z, iteration, colormap, and contrast in real time. Export individual ion images as high-resolution SVG/PNG or dump the full datacube as calibrated TIFFs (with embedded resolution metadata for ImageJ/Fiji).

## 🔧 Key Features

✨ **Realistic PSF Simulation** (`src/psf_generator.py`)

The PSF is simulated on a fine grid (0.05 µm resolution) and then integrated down to the coarse pixel grid. Two beam models are supported:

| Model | Description |
|---|---|
| `beam_profile` | Continuous **Gaussian** beam ($I(r) = e^{-2r^2/w_0^2}$). Captures subtle, spatially-varying signal degradation. |
| `geometric` | Binary **circular** aperture. Counts discrete shot overlaps. Faster, useful as a baseline. |

Signal degradation from repeated laser exposure is modeled via the **decay profile**, with two strategies:

| Decay Model | Description |
|---|---|
| `exponential` | Fits $f(n) = a \cdot e^{-b \cdot n} + c$ to measured per-pass intensities. Robust and physically motivated. |
| `interpolation` | Linear interpolation between measured points. No assumptions on decay shape. |

⚡ **Efficient Processing**
- Dask-parallelized Richardson-Lucy deconvolution
- Batch processing of full ion image stacks
- Minimal memory footprint for large datasets

📊 **Data Structure & Interoperability**
- **xarray** for labeled N-dimensional data (preserves metadata, coordinates, dimensions)
- Seamless integration with scientific Python ecosystem
- NetCDF (`.h5`) format for portability and reproducibility

📈 **Interactive Visualization**
- **HoloViews** for rapid, responsive real-time previews in notebooks
- Hover tooltips, zoom, pan for exploratory analysis
- Dual backends (Matplotlib + Bokeh)

🎨 **Publication-Ready Output**
- High-resolution plots (SVG/PNG)
- Palette-mapped TIFF export compatible with ImageJ/Fiji
- Batch export of full datasets with embedded resolution metadata

## 💡 Usage Tips

- Adjust PSF parameters (laser diameter, step size) to match your instrument specs
- The decay profile CSV can be produced by notebook 2 or provided manually (1 or 2 columns, with or without header)
- Inspect each deconvolution iteration in notebook 4 to choose the optimal stopping point
- Use `beam_profile` + `exponential` for the most physically accurate results

## 📁 Project Structure

```
├── notebooks/                      # 📓 Interactive marimo notebooks
│   ├── 1a_extract_data.py
│   ├── 1b_extract_decay_profile.py
│   ├── 2_deconvolve.py
│   └── 3_visualize.py
├── src/
│   └── psf_generator.py            # 🔧 PSF simulation engine
├── data/                           # 📂 Input datasets
├── pyproject.toml                  # 📦 Project dependencies
├── uv.lock                         # 🔒 Locked dependency versions
└── README.md                       # 📖 This file
```

## 📚 Citation

TBD

## 🤝 Contributing

Contributions welcome! Feel free to open issues or PRs.

## CLI Usage
### `1a_extract_data.py`
```bash
uv run notebooks/1a_extract_data.py -- --input PATH [--mz VALUE | --mz-csv PATH] [--tolerance VALUE]
```

| Argument | Required | Description |
| --- | --- | --- |
| `--input PATH` | yes | Path to the `.imzml` file to process |
| `--mz VALUE` | no* | Single m/z value to extract |
| `--mz-csv PATH` | no* | Path to a CSV with m/z values (no header, one value per line) |
| `--tolerance VALUE` | no | m/z tolerance in Da (default: `0.005`) |

\* If neither `--mz` nor `--mz-csv` is provided, the midpoint of the m/z range is used.

---

**Happy deconvolving!** 🔬✨
