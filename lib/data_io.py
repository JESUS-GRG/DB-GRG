# lib/data_io.py
import os, json
import requests
import pandas as pd
from .utils import (
    MONTH_LABELS, LABEL2NUM, MONTH_ORDER, MONTH_ORDER_CAP,
    norm_series_noaccents, apply_state_map, parse_month, classify_equipment_type
)

# Carga CURATED con derivadas mínimas y normalizaciones UNA sola vez
def load_curated(curated_path: str) -> pd.DataFrame:
    if not os.path.exists(curated_path):
        raise FileNotFoundError(f"Curated parquet not found: {curated_path}")
    df = pd.read_parquet(curated_path, engine="pyarrow")

    # Derivadas clave:
    # Year
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    # Month -> entero robusto
    if "Month" in df.columns:
        df["_mnum"] = df["Month"].map(parse_month).astype("Int64")
    else:
        df["_mnum"] = pd.Series(dtype="Int64")

    # ServiceNorm
    if "Service Type" in df.columns:
        df["ServiceNorm"] = norm_series_noaccents(df["Service Type"])
    else:
        df["ServiceNorm"] = "n/a"

    # SparePartFlag
    spare_col = "Solved By Using Spare Part"
    if spare_col in df.columns:
        def _spare(v):
            if pd.isna(v): return "BLANK"
            s = str(v).strip().upper()
            return s if s in ("YES", "NO") else "BLANK"
        df["SparePartFlag"] = df[spare_col].map(_spare)
    else:
        df["SparePartFlag"] = "BLANK"

    # Estado/Región normalizados
    if "State" in df.columns:
        df["_state_clean"] = apply_state_map(df["State"])
        df["_state_norm"]  = norm_series_noaccents(df["_state_clean"])
    if "Region" in df.columns:
        df["_region_norm"] = norm_series_noaccents(df["Region"])

    return df

# Carga BDI y estandariza campos
def load_bdi(raw_bdi_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_bdi_path):
        return pd.DataFrame()
    bdi = pd.read_parquet(raw_bdi_path, engine="pyarrow").copy()

    # Normaliza columnas comunes
    if "Estado" in bdi.columns and "State" not in bdi.columns:
        bdi["State"] = bdi["Estado"].astype("string")
    if "Ciudad" in bdi.columns and "City" not in bdi.columns:
        bdi["City"] = bdi["Ciudad"].astype("string")

    # Year
    if "Year" in bdi.columns:
        bdi["Year"] = pd.to_numeric(bdi["Year"], errors="coerce").astype("Int64")

    # Month -> _mnum
    if "Month" in bdi.columns:
        bdi["Month"] = bdi["Month"].astype(str).str.strip()
        bdi["_mnum"] = bdi["Month"].map(parse_month).astype("Int64")
    else:
        bdi["_mnum"] = pd.Series(dtype="Int64")

    # Equipment type → Class2
    eqcol = "Equipment Type" if "Equipment Type" in bdi.columns else None
    if eqcol:
        bdi["Class2"] = bdi[eqcol].map(classify_equipment_type)
    else:
        bdi["Class2"] = "Other"

    # Región estándar (Metro Norte/Sur → Metro)
    if "Region" in bdi.columns:
        bdi["RegionStd"] = (
            bdi["Region"].astype("string").str.strip()
            .replace({"Metro Norte": "Metro", "Metro Sur": "Metro"})
        )
    else:
        bdi["RegionStd"] = pd.Series(dtype="string")

    return bdi

# GEOJSON helpers
def ensure_geojson(primary_path: str, alt_path: str|None=None) -> str|None:
    """Devuelve una ruta existente a GeoJSON. Si no existe y hay internet, intenta descargar mexicoHigh.json.
    Si nada existe y falla descarga: retorna None (la app mostrará aviso)."""
    if os.path.exists(primary_path):
        return primary_path
    if alt_path and os.path.exists(alt_path):
        return alt_path
    # intento de descarga
    try:
        os.makedirs(os.path.dirname(primary_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(primary_path, "wb") as f:
            f.write(r.content)
        return primary_path
    except Exception:
        return None

def load_geojson(path: str|None):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    candidates = ["name", "NOMGEO", "nom_ent", "NOM_ENT", "estado", "state_name", "NAME_1"]
    feats = gj.get("features", [])
    counts = {}
    for feat in feats:
        props = feat.get("properties", {})
        for k in candidates:
            if props.get(k):
                counts[k] = counts.get(k, 0) + 1
    name_field = max(counts, key=counts.get) if counts else "name"

    # Add name_norm
    import unicodedata, re
    def _norm(s):
        s = str(s).strip().lower()
        s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
        s = re.sub(r'[_\-/]+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s
    for feat in feats:
        props = feat.get("properties", {})
        raw = props.get(name_field, "")
        props["name_norm"] = _norm(raw)
        feat["properties"] = props
    gj["_detected_name_field"] = name_field
    return gj
