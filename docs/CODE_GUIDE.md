# HistoRAG — Code Guide

Concise descriptions of every source file. Focus: *what each block does and why*, not line-by-line mechanics.

---

## `histoRAG/tile.py` — WSI Loading and Patch Extraction

**Purpose**: Open a whole-slide image (WSI), find the tissue, cut it into small square patches, and save those patches as PNG files with a metadata record for each.

A WSI is a gigapixel image stored as a multi-resolution pyramid (think Google Maps for microscopy). You never load the whole image at once; instead you pick a zoom level and read small regions on demand.

---

### `class WSI` — Thin Slide Wrapper

Wraps OpenSlide (the standard library for reading WSI formats: `.svs`, `.ndpi`, `.tiff`, etc.).

**What it does**:
- Opens a slide file and exposes its pyramid levels.
- `best_level_for_mag(target_mag)`: the pyramid stores the slide at several downsampled resolutions. This method reads the slide's objective-power metadata, calculates the effective magnification at each level, and returns the level closest to your target (e.g. 20×). If metadata is missing it falls back gracefully to level 0.
- `read_region_rgb()` / `get_thumbnail()`: thin wrappers around OpenSlide calls that convert the RGBA output to RGB.
- Supports Python's `with` statement so the file handle is always closed.

---

### `_otsu_threshold(channel)` — Tissue/Background Separator

Implements Otsu's algorithm from scratch on a 2-D grayscale array.

**What it does**: Finds the pixel intensity value that best splits a histogram into two groups (tissue vs. background) by maximizing the between-class variance. Returns a single threshold number. No external library needed — the loop walks all 256 possible thresholds and picks the one with the highest separability score.

**Why it exists here**: Applied to the HSV *saturation* channel of the thumbnail. Tissue is colourful (high saturation); glass/whitespace is pale (low saturation). Otsu on saturation is a robust, parameter-free way to produce a binary tissue mask.

---

### `class Tiler` — Grid Extractor with Tissue Filter

The main workhorse. Takes a WSI and produces a list of patch records.

**Configuration parameters** (set at construction, come from `configs/phase0_mvp.yaml`):
- `patch_size` / `stride`: grid step size in pixels at the chosen pyramid level.
- `target_magnification`: selects which pyramid level to extract from.
- `thumb_downsample`: how much to shrink the slide for computing the tissue mask (doing it at full resolution would be too slow).
- `min_tissue_frac`: a patch is kept only if at least this fraction of its area overlaps with tissue in the mask.
- `max_patches_per_slide`: cap to avoid memory/storage explosion; excess patches are randomly subsampled.

**`extract(wsi, slide_id, out_dir, label)` — step-by-step**:

1. **Select pyramid level** via `best_level_for_mag`.
2. **Build tissue mask**: get a small thumbnail, convert it to HSV colour space, run Otsu on the saturation channel → binary mask (True = tissue).
3. **Grid scan**: iterate over all possible (x, y) patch positions at the chosen level. For each candidate, project its bounding box into the thumbnail's coordinate space and compute what fraction of those thumbnail pixels are tissue. Discard if below `min_tissue_frac`.
4. **Cap**: if surviving patches exceed `max_patches_per_slide`, randomly subsample using a seeded RNG (deterministic).
5. **Extract and save**: for each kept patch, call `read_region_rgb` on the actual full-resolution pyramid level and save as a PNG.
6. **Return manifest rows**: each row is a dict with `patch_id`, coordinates, pyramid level, effective magnification, label, and file path.

**Output**: list of dicts consumed by `pipeline.py` to build `manifest.parquet`.

---

## `histoRAG/embed.py` — CLIP Encoding and FAISS Indexing

**Purpose**: Convert patch images into fixed-size numerical vectors (embeddings) that capture visual meaning, and store/search those vectors efficiently.

The core idea: CLIP was trained on 400 million image–text pairs from the internet. Its image encoder learned to produce vectors where visually similar images are numerically close — without any histopathology-specific training. We exploit this for retrieval.

---

### `class ClipEncoder` — Image → Vector

Wraps the `open_clip` library to load CLIP ViT-B/16 with OpenAI weights.

**`__init__`**: Loads the model and its built-in preprocessing pipeline (resize to 224×224, normalize with CLIP's mean/std). Moves model to GPU if available.

**`encode(images)`**:
1. Stack PIL images into a batch tensor after applying the preprocessing pipeline.
2. Pass through the CLIP vision transformer (no text tower used here).
3. **L2-normalize** the output vectors: divide each vector by its own length so it becomes a unit vector. After normalization, the inner product (dot product) between two vectors equals their cosine similarity. This is the mathematical trick that lets us use a dot-product index for cosine search.
4. Return a `(N, 512)` float32 NumPy array.

**`encode_batched(images, batch_size)`**: Calls `encode()` in chunks with a tqdm progress bar. Needed because loading all patches into GPU memory at once would cause OOM.

---

### `class FaissFlatIP` — Exact Vector Search

Wraps FAISS `IndexFlatIP` (Flat = brute-force, IP = inner product / dot product).

**`__init__`**: Creates an `IndexIDMap2` wrapping `IndexFlatIP`. The outer `IDMap2` layer maps arbitrary integer patch IDs to FAISS's internal 0-based row indices — so you get back your original patch IDs from a search, not FAISS row numbers.

**`add(embeddings, ids)`**: Inserts all embedding vectors with their associated patch IDs. Converts to contiguous float32 and int64 arrays as required by FAISS.

**`search(queries, k)`**: Computes exact dot products between each query vector and every stored vector. Returns the top-k results as `(similarities, ids)` arrays of shape `(Q, k)`. Since all vectors are L2-normalized, these similarities are cosine similarities in [−1, 1].

**`save` / `load`**: Serializes/deserializes the entire index to a `.faiss` file so you don't have to re-encode on every run.

**Why "Flat" (brute-force)?** At Phase 0 scale (≤100 k vectors, 512 dimensions), brute-force is fast enough (milliseconds per query) and gives **exact** answers. Approximate methods (IVF, HNSW) sacrifice accuracy for speed — reserved as Phase 2 ablations.

---
