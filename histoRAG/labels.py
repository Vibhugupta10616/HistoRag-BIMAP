"""
Label derivation utilities for H1 and H2 hypotheses.

Provides two functions:
  - tumour_labels_from_geojson: assigns 'tumour' or 'other' to each patch
    using sparse QuPath polygon annotations (.geojson files, one per slide).
  - site_labels_from_clinical: assigns anatomical site (oropharynx / larynx /
    oral_cavity) to each patch from HANCOCK per-patient clinical metadata.

NOTE: Both functions require HANCOCK data that is not yet downloaded.
They are written against the documented formats; confirm exact column names
and file layout when the real dataset arrives.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Tumour labels from QuPath .geojson annotations
# ---------------------------------------------------------------------------

def tumour_labels_from_geojson(
    manifest: pd.DataFrame,
    geojson_dir: str | Path,
    patch_size: int = 256,
) -> pd.Series:
    """
    Assign 'tumour' or 'other' to every patch in the manifest.

    A patch is labelled 'tumour' if its center (x, y) falls inside any
    polygon in that slide's .geojson annotation file.  Everything outside
    is 'other' because HANCOCK only annotates tumour regions (sparse
    annotation — non-tumour tissue is unannotated, not a separate class).

    Args:
        manifest:    patch manifest DataFrame with columns patch_id,
                     slide_id, x, y (x/y are top-left corner coords in
                     WSI pixel space at the extraction level).
        geojson_dir: directory containing one .geojson file per slide,
                     named <slide_id>.geojson (QuPath export format).
        patch_size:  patch width/height in WSI pixels. Used to convert
                     top-left coordinates to patch centers.

    Returns:
        pd.Series of str, index aligned to manifest.index,
        values are 'tumour' or 'other'.
    """
    geojson_dir = Path(geojson_dir)

    # Build a flat index {slide_id: path} by scanning recursively.
    # This handles both flat layouts (local: geojson_dir/*.geojson) and
    # nested layouts (HPC: geojson_dir/<subdir>/*.geojson).
    geojson_index: dict[str, Path] = {
        p.stem: p for p in geojson_dir.rglob("*.geojson")
    }
    n_slides = manifest["slide_id"].nunique()
    missing = [sid for sid in manifest["slide_id"].unique() if sid not in geojson_index]
    print(f"[labels] Geojson index: {len(geojson_index)} files found under {geojson_dir}")
    if missing:
        print(f"[labels] Warning: {len(missing)}/{n_slides} slides have no geojson "
              f"— those patches will be labeled 'other'. First missing: {missing[:3]}")
    if len(missing) == n_slides:
        raise FileNotFoundError(
            f"No geojson files matched any of the {n_slides} slides.\n"
            f"  Slide IDs (first 3): {list(manifest['slide_id'].unique())[:3]}\n"
            f"  Geojson files found (first 3): {list(geojson_index.keys())[:3]}\n"
            f"  Searched under: {geojson_dir}\n"
            "Check that geojson_dir is correct and files are named <slide_id>.geojson."
        )

    labels = pd.Series("other", index=manifest.index, dtype=str)

    for slide_id, group in manifest.groupby("slide_id"):
        geojson_path = geojson_index.get(slide_id)
        if geojson_path is None:
            continue

        polygons = _load_tumour_polygons(geojson_path)
        if not polygons:
            print(f"[labels] No tumour polygons found in {geojson_path.name}.")
            continue

        centers = group[["x", "y"]].values.astype(float) + (patch_size / 2.0)

        # a patch is tumour if it falls inside ANY polygon for this slide
        is_tumour = np.zeros(len(group), dtype=bool)
        for poly_coords in polygons:
            path = MplPath(poly_coords)
            is_tumour |= path.contains_points(centers)

        labels.loc[group.index[is_tumour]] = "tumour"

    n_tumour = (labels == "tumour").sum()
    print(f"[labels] Tumour patches: {n_tumour} / {len(labels)} "
          f"({100 * n_tumour / max(len(labels), 1):.1f}%)")
    return labels


def _load_tumour_polygons(geojson_path: Path) -> list[list[tuple]]:
    """
    Parse a QuPath .geojson file and return polygon coordinate lists.

    QuPath stores annotations as GeoJSON FeatureCollections.  Each Feature
    has a geometry of type Polygon or MultiPolygon.  We extract the outer
    ring of each polygon (holes are ignored at this stage).

    Returns:
        List of polygons, where each polygon is a list of (x, y) tuples.
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        geom_type = geom.get("type", "")
        coords_raw = geom.get("coordinates", [])

        if geom_type == "Polygon":
            # coords_raw = [[[x, y], ...], ...]  — outer ring is index 0
            outer_ring = coords_raw[0]
            polygons.append([tuple(pt[:2]) for pt in outer_ring])

        elif geom_type == "MultiPolygon":
            # coords_raw = [[[[x, y], ...]], ...]  — list of polygons
            for poly in coords_raw:
                outer_ring = poly[0]
                polygons.append([tuple(pt[:2]) for pt in outer_ring])

    return polygons


# ---------------------------------------------------------------------------
# Anatomical site labels from HANCOCK clinical metadata
# ---------------------------------------------------------------------------

def site_labels_from_clinical(
    manifest: pd.DataFrame,
    clinical_csv: str | Path,
    slide_id_col: str = "patient_id",
    site_col: str = "primary_site",
) -> pd.Series:
    """
    Assign the anatomical primary site to every patch from clinical metadata.

    HANCOCK provides one anatomical site label per patient (oropharynx /
    larynx / oral cavity) in a clinical metadata CSV.  Because each slide
    belongs to one patient, this becomes a per-patch label via the slide_id.

    Args:
        manifest:      patch manifest with column 'slide_id'.
        clinical_csv:  path to HANCOCK clinical metadata CSV.
        slide_id_col:  column name in the CSV that matches manifest.slide_id.
                       Default 'patient_id'; adjust to the real column name
                       after downloading the dataset.
        site_col:      column name in the CSV containing the site label.
                       Default 'primary_site'; adjust as needed.

    Returns:
        pd.Series of str, index aligned to manifest.index.
        Patches with no matching clinical record receive 'unknown'.
    """
    clinical_csv = Path(clinical_csv)
    if not clinical_csv.exists():
        raise FileNotFoundError(
            f"Clinical metadata file not found: {clinical_csv}\n"
            "Download the HANCOCK dataset and set the correct path in config.yaml."
        )

    clinical = pd.read_csv(clinical_csv)

    # build a slide_id → site mapping
    site_map: dict[str, str] = dict(
        zip(clinical[slide_id_col].astype(str), clinical[site_col].astype(str))
    )

    site_labels = manifest["slide_id"].astype(str).map(site_map).fillna("unknown")
    site_labels.index = manifest.index

    n_unknown = (site_labels == "unknown").sum()
    if n_unknown > 0:
        print(
            f"[labels] Warning: {n_unknown} patches have no matching clinical record "
            f"(slide_id_col='{slide_id_col}'). They are labelled 'unknown'."
        )

    print(f"[labels] Site distribution: {dict(site_labels.value_counts())}")
    return site_labels
