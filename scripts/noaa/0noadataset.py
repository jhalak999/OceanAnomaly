import pandas as pd
import xarray as xr
from pathlib import Path
import requests
import re

# ================= USER CONTROL =================
START_YEAR = 1981
END_YEAR = 2024
# ===============================================

out_dir = Path("noaa/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

url = "https://psl.noaa.gov/data/correlation/oni.data"

print(f"📡 Extracting NOAA ONI data for years {START_YEAR}–{END_YEAR}")

# ------------------------------------------------
# 1. DOWNLOAD FILE AS TEXT
# ------------------------------------------------
text = requests.get(url, timeout=30).text
lines = text.splitlines()

# ------------------------------------------------
# 2. MANUALLY PARSE VALID DATA ROWS
# ------------------------------------------------
records = []

for line in lines:
    # keep only lines starting with 4-digit year
    if re.match(r"^\s*\d{4}\s+", line):
        parts = line.split()
        if len(parts) == 13:   # Year + 12 seasons
            year = int(parts[0])
            if START_YEAR <= year <= END_YEAR:
                records.append(parts)

# safety check
if not records:
    raise RuntimeError("No valid ONI records found after parsing")

# ------------------------------------------------
# 3. CREATE DATAFRAME
# ------------------------------------------------
columns = [
    "YR", "DJF", "JFM", "FMA", "MAM", "AMJ",
    "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"
]

oni = pd.DataFrame(records, columns=columns).astype(float)
oni["YR"] = oni["YR"].astype(int)

print(f"✅ Years retained: {oni['YR'].min()} – {oni['YR'].max()}")

# ------------------------------------------------
# 4. SEASON → MONTH
# ------------------------------------------------
season_map = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
}

oni_long = oni.melt(
    id_vars="YR",
    var_name="season",
    value_name="oni"
)

oni_long["month"] = oni_long["season"].map(season_map)

oni_long["time"] = pd.to_datetime(
    oni_long["YR"].astype(str) + "-" +
    oni_long["month"].astype(str) + "-01",
    format="%Y-%m-%d"
)

oni_long = oni_long.sort_values("time")

# ------------------------------------------------
# 5. SAVE OUTPUTS
# ------------------------------------------------
oni_long[["time", "oni"]].to_csv(
    out_dir / "noaa_oni_monthly.csv",
    index=False
)

ds = xr.Dataset(
    {"oni": (["time"], oni_long["oni"].values)},
    coords={"time": oni_long["time"].values}
)

ds.to_netcdf(out_dir / "noaa_oni_monthly.nc")

print("🎉 NOAA ONI extraction complete (manual parsing, robust)")
