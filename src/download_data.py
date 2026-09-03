"""
Download the Lending Club dataset from Kaggle.

Dataset: wordsforthewise/lending-club
  - accepted_2007_to_2018Q4.csv.gz  (~2.2M funded loans, 151 cols) <- what we model
  - rejected_2007_to_2018Q4.csv.gz  (declined applications; not used yet)

Prerequisite: a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600).
Get one at https://www.kaggle.com/settings  ->  "Create New Token".

Usage:
    python src/download_data.py            # downloads accepted loans only
    python src/download_data.py --rejected # also grab the rejected file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET = "wordsforthewise/lending-club"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ACCEPTED_FILE = "accepted_2007_to_2018Q4.csv.gz"
REJECTED_FILE = "rejected_2007_to_2018Q4.csv.gz"


AUTH_HELP = (
    "\nKaggle authentication failed. Set up ONE of these (all handled by the client):\n"
    "  * OAuth (recommended): ./venv/bin/kaggle auth login  "
    "-> caches ~/.kaggle/credentials.json\n"
    "  * Token file: create at https://www.kaggle.com/settings -> API -> "
    "Create New Token,\n"
    "      then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && "
    "chmod 600 ~/.kaggle/kaggle.json\n"
    "  * Env var: export KAGGLE_USERNAME=... KAGGLE_KEY=...\n"
    "Verify with: ./venv/bin/kaggle datasets list -s lending-club | head\n"
)


def _authenticate():
    """Authenticate however the user set it up (OAuth, token file, or env var).

    We don't pre-check for a specific credential file -- the Kaggle client
    supports several methods (e.g. OAuth writes credentials.json, not
    kaggle.json). We just try, and surface help only if it genuinely fails.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - want a friendly message for any auth failure
        sys.exit(f"{AUTH_HELP}\n(underlying error: {exc})\n")
    return api


def download(files: list[str]) -> None:
    api = _authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in files:
        print(f"Downloading {fname} -> {RAW_DIR} (this is large; be patient)...")
        api.dataset_download_file(
            DATASET, file_name=fname, path=str(RAW_DIR), force=False, quiet=False
        )
    print("\nDone. Files in data/raw/:")
    for p in sorted(RAW_DIR.glob("*.csv.gz")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rejected", action="store_true", help="also download the rejected-loans file"
    )
    args = ap.parse_args()

    wanted = [ACCEPTED_FILE]
    if args.rejected:
        wanted.append(REJECTED_FILE)
    download(wanted)
