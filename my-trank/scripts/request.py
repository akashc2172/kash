from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
CLEAN_DIR = REPO_ROOT / "data" / "cleaned_csvs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

print("SCRIPT PATH:", Path(__file__).resolve())
print("REPO_ROOT   :", REPO_ROOT)
print("RAW_DIR     :", RAW_DIR)
print("CLEAN_DIR   :", CLEAN_DIR)
print("RAW exists? :", RAW_DIR.exists())
print("CLEAN exists:", CLEAN_DIR.exists())


import time
import random
import json
from pathlib import Path

import requests
import pandas as pd

# =========================
# DOWNLOAD CONFIG
# =========================
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://hoop-explorer.com/",
}

# Static JSON directory (consistent across years)
STATIC_DIR = "https://hoop-explorer.com/leaderboards/lineups"

# API endpoint (backup)
API_BASE = "https://hoop-explorer.com/api/getLeaderboard"
API_PARAMS_TEMPLATE = {
    "src": "players",
    "oppo": "all",
    "gender": "Men",
    "type": "player",
}

# =========================
# CLEANING CONFIG
# =========================
INPUT_DIR = RAW_DIR
OUTPUT_DIR = CLEAN_DIR
INPUT_GLOB = "players_*_HML.csv"
REFERENCE_CSV = "players_2025_HML.csv"


# If True: skip files that have ANY extra cols (not in final common schema)
# If False: just drop extras and keep the intersection schema.
SKIP_FILES_WITH_EXTRA_COLS = False

# Optional: enforce some columns must exist in every output (under intersection rule)
REQUIRED_COLS = ["_id", "year", "tier", "gender"]


# =========================
# DOWNLOAD FUNCTIONS
# =========================
def fetch_static(year: int, tier: str, gender: str = "Men") -> pd.DataFrame | None:
    fname = f"players_all_{gender}_{year}_{tier}.json"
    url = f"{STATIC_DIR}/{fname}"

    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    j = r.json()
    rows = j["players"] if isinstance(j, dict) and "players" in j else j

    df = pd.json_normalize(rows)
    df["year"] = year
    df["tier"] = tier
    df["gender"] = gender
    return df


def fetch_api(year: int, tier: str, gender: str = "Men") -> pd.DataFrame | None:
    params = dict(API_PARAMS_TEMPLATE)
    params.update({"year": str(year), "tier": tier, "gender": gender})

    r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    j = r.json()
    if "error" in j:
        return None

    df = pd.json_normalize(j["players"])
    df["year"] = year
    df["tier"] = tier
    df["gender"] = gender
    return df


def get_players(year: int, tier: str, gender: str = "Men") -> pd.DataFrame | None:
    # SAME approach for all years:
    # 1) try static json (works for old + often new)
    # 2) if missing, fall back to API
    df = fetch_static(year, tier, gender=gender)
    if df is not None and not df.empty:
        return df
    return fetch_api(year, tier, gender=gender)


def download_year_files(start_year: int = 2018, end_year: int = 2025, gender: str = "Men") -> None:
    print("[download] writing RAW csvs to:", RAW_DIR.resolve())

    tiers = ["High", "Medium", "Low"]
    all_years = []

    for year in range(start_year, end_year + 1):
        year_dfs = []
        for tier in tiers:
            print(f"[download] Fetching year={year} tier={tier}")
            df = get_players(year, tier, gender=gender)

            if df is None or df.empty:
                print(f"  -> missing (404/empty): year={year} tier={tier}")
                continue

            # dedupe within tier fetch
            if "_id" in df.columns:
                df = df.drop_duplicates(subset=["_id"], keep="first")

            year_dfs.append(df)
            time.sleep(random.uniform(0.3, 0.9))

        if not year_dfs:
            print(f"[download] Year {year}: no tiers available")
            continue

        df_year = pd.concat(year_dfs, ignore_index=True)

        # prevent cross-tier duplicates within a year
        if "_id" in df_year.columns:
            df_year = df_year.drop_duplicates(subset=["year", "_id"], keep="first")

        df_year.to_csv(RAW_DIR / f"players_{year}_HML.csv", index=False)
        print(f"[download] Saved players_{year}_HML.csv shape={df_year.shape}")
        all_years.append(df_year)

    if all_years:
        big = pd.concat(all_years, ignore_index=True)

        # prevent accidental cross-year collapse
        if "_id" in big.columns:
            big = big.drop_duplicates(subset=["year", "_id"], keep="first")

        big.to_csv(RAW_DIR / f"players_{start_year}_{end_year}_HML.csv", index=False)
        print(f"[download] Saved players_{start_year}_{end_year}_HML.csv shape={big.shape}")


# =========================
# CLEANING FUNCTIONS
# =========================
def clean_all_year_csvs(

    input_dir: Path = INPUT_DIR,
    input_glob: str = INPUT_GLOB,
    output_dir: Path = OUTPUT_DIR,
    reference_csv: str | None = REFERENCE_CSV,
    skip_files_with_extra_cols: bool = SKIP_FILES_WITH_EXTRA_COLS,
    required_cols: list[str] | None = None,
) -> None:
    print("[clean] reading from:", input_dir.resolve())
    print("[clean] writing to  :", output_dir.resolve())

    if required_cols is None:
        required_cols = []

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(input_glob))
    if not files:
        print("[clean] glob pattern:", str(input_dir / input_glob))
        print("[clean] files present:", [p.name for p in input_dir.glob("*")][:20])
        raise FileNotFoundError(f"No files matched {input_dir / input_glob}")

    # choose reference file for ordering
    if reference_csv is None:
        ref_path = files[0]
    else:
        ref_path = input_dir / reference_csv
        if not ref_path.exists():
            # fall back to first file if the named ref isn't present
            ref_path = files[0]

    # read headers only
    headers = {str(f): list(pd.read_csv(f, nrows=0).columns) for f in files}

    # intersection: columns present in EVERY file
    common_cols = set(headers[str(files[0])])
    for f in files[1:]:
        common_cols &= set(headers[str(f)])
    common_cols = list(common_cols)

    # sanity check required columns
    missing_required = [c for c in required_cols if c not in common_cols]
    if missing_required:
        raise ValueError(
            "These required columns are not present in every file, so they cannot be kept under "
            f"intersection schema: {missing_required}"
        )

    # order columns: reference order first, then leftover common cols sorted
    ref_cols = list(pd.read_csv(ref_path, nrows=0).columns)
    ordered_common = [c for c in ref_cols if c in common_cols]
    leftover = sorted([c for c in common_cols if c not in set(ordered_common)])
    final_cols = ordered_common + leftover

    # write schema file (the exact final order)
    (output_dir / "common_columns_in_order.txt").write_text(
        "\n".join(final_cols) + "\n", encoding="utf-8"
    )

    dropped_report = {}
    skipped_files = []

    final_set = set(final_cols)

    for f in files:
        cols = headers[str(f)]
        extra_cols = [c for c in cols if c not in final_set]

        if skip_files_with_extra_cols and extra_cols:
            skipped_files.append(str(f))
            continue

        # Read only the final columns, in final order
        df = pd.read_csv(f, usecols=final_cols, low_memory=False)
        df = df.reindex(columns=final_cols)

        out_path = output_dir / f.name
        df.to_csv(out_path, index=False)

        dropped_report[f.name] = {
            "dropped_columns_count": len(extra_cols),
            "dropped_columns": extra_cols,
            "kept_columns_count": len(final_cols),
        }

    (output_dir / "dropped_columns_by_file.json").write_text(
        json.dumps(dropped_report, indent=2), encoding="utf-8"
    )

    if skipped_files:
        (output_dir / "skipped_files.txt").write_text("\n".join(skipped_files) + "\n", encoding="utf-8")

    print(f"[clean] Found {len(files)} files")
    print(f"[clean] Common columns kept: {len(final_cols)}")
    print(f"[clean] Wrote cleaned CSVs to: {output_dir.resolve()}")
    if skipped_files:
        print(f"[clean] Skipped {len(skipped_files)} files (skip_files_with_extra_cols=True)")


# =========================
# RUN EVERYTHING
# =========================
if __name__ == "__main__":
    # 1) download / refresh raw year files
    download_year_files(start_year=2018, end_year=2025, gender="Men")
    print("[debug] RAW files now:", len(list(RAW_DIR.glob("players_*_HML.csv"))))

    # 2) clean them to a common schema + consistent column order
    clean_all_year_csvs(
        input_dir=INPUT_DIR,
        input_glob=INPUT_GLOB,
        output_dir=OUTPUT_DIR,
        reference_csv=REFERENCE_CSV,
        skip_files_with_extra_cols=SKIP_FILES_WITH_EXTRA_COLS,
        required_cols=REQUIRED_COLS,
    )
