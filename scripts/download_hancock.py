"""Download HANCOCK WSI slides listed in the config.

Reads slide_ids from configs/phase0_mvp.yaml and downloads each slide
to data/raw/<slide_id>.<ext>.

NOTE: Actual download URLs depend on the HANCOCK portal authentication.
Update HANCOCK_BASE_URL and auth logic below after logging in at
www.hancock.research.fau.eu.

Usage:
    python scripts/download_hancock.py --config configs/phase0_mvp.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from histoRAG.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Download HANCOCK slides")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    slide_ids = cfg["data"].get("slide_ids", [])
    print(f"Slides to download: {slide_ids}")
    print(f"Target directory  : {raw_dir}")
    print()

    # ── Configure this section after registering at hancock.research.fau.eu ──
    HANCOCK_BASE_URL = "https://hancock.research.fau.eu/downloads"  # placeholder
    # token = os.environ.get("HANCOCK_TOKEN") or input("HANCOCK token: ")
    # headers = {"Authorization": f"Bearer {token}"}

    for slide_id in slide_ids:
        if slide_id.startswith("TBD"):
            print(f"  SKIP {slide_id} — placeholder ID, update configs/phase0_mvp.yaml")
            continue

        # Attempt download (update URL pattern to match HANCOCK portal)
        for ext in [".svs", ".tiff", ".ndpi"]:
            dest = raw_dir / f"{slide_id}{ext}"
            if dest.exists():
                print(f"  ALREADY EXISTS: {dest}")
                break
            url = f"{HANCOCK_BASE_URL}/{slide_id}{ext}"
            print(f"  Downloading {url} → {dest}")
            # Uncomment once URL pattern is confirmed:
            # import urllib.request
            # urllib.request.urlretrieve(url, dest)
            # OR use requests with auth:
            # import requests
            # r = requests.get(url, headers=headers, stream=True)
            # with open(dest, "wb") as f:
            #     for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print("  (download logic commented out — fill in URL + auth above)")
            break


if __name__ == "__main__":
    main()
