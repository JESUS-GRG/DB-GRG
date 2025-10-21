# lib/utils.py
import unicodedata
import re
import pandas as pd

# --- Month labels/maps (bidireccional y robusto) ---
MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
LABEL2NUM = {v:k for k, v in MONTH_LABELS.items()}
MONTH_ORDER = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
MONTH_ORDER_CAP = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
DAY_ORDER = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

# Estados comunes mal escritos/variantes → forma estándar
STATE_MAP_RAW = {
    "Edo. De México": "México",
    "Estado De Mexico": "México",
    "Estado de México": "México",
    "CDMX": "Ciudad de México",
    "Distrito Federal": "Ciudad de México",
    "Nuevo Leon": "Nuevo León",
    "San Luis Potosi": "San Luis Potosí",
    "Michoacan": "Michoacán",
    "Yucatan": "Yucatán",
    "Queretaro": "Querétaro",
    # agrega aquí si detectas más variantes en tus datos
}

# Paletas
QUAL_PALETTE = [
    # Plotly + T10 combinadas (subset fijo)
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
    "#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#4c78a8","#f58518","#54a24b","#e45756","#72b7b2","#f2cf5b"
]

RED_PALETTE = ["#b30000", "#e34a33", "#fc8d59", "#fdbb84", "#99000d", "#cb181d", "#fb6a4a"]

def _strip_accents_lower(s: str) -> str:
    s = str(s).strip().lower()
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    s = re.sub(r'[_\-/]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def norm_series_noaccents(sr: pd.Series) -> pd.Series:
    return sr.astype(str).map(_strip_accents_lower)

def apply_state_map(sr: pd.Series) -> pd.Series:
    """Regresa nombres estándar de estado con mapeo de variantes comunes."""
    norm_map = {_strip_accents_lower(k): v for k, v in STATE_MAP_RAW.items()}
    keys = sr.astype(str).map(_strip_accents_lower)
    mapped = keys.map(norm_map)
    out = sr.astype(str).copy()
    out.loc[mapped.notna()] = mapped.loc[mapped.notna()]
    return out

def pct(n, d):
    if d in (0, None) or pd.isna(d): return 0.0
    return 100.0 * (n / d)

def fmt_pct(x, decimals=1):
    return f"{x:.{decimals}f}%"

def safe_default(options, defaults):
    opt_set = set(list(options))
    return [d for d in list(defaults) if d in opt_set]

def year_palette(years):
    base = ["#0096c7", "#023e8a", "#2a9d8f", "#e76f51", "#f4a261", "#6a4c93"]
    years = [str(int(y)) for y in sorted({int(y) for y in pd.Series(list(years)).dropna().astype(int).tolist()})]
    cmap = {years[i]: base[i % len(base)] for i in range(len(years))}
    return years, cmap

# --- Mes robusto: acepta 1-12, "1", "01", "Jan", "JAN", "Sept", "September" ---
_MONTH_ALIASES = {
    "JAN":1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6, "JUL":7, "AUG":8, "SEP":9, "SEPT":9, "OCT":10, "NOV":11, "DEC":12
}
def parse_month(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    # numérico directo
    if s.isdigit():
        try:
            m = int(s)
            return m if 1 <= m <= 12 else None
        except Exception:
            return None
    # texto
    s_up = _strip_accents_lower(s).upper()
    s_up = re.sub(r'[^A-Z]', '', s_up)
    # ejemplos: "JAN", "SEPTEMBER" -> "SEPTEMBER"
    # map por prefijo común
    for k, v in _MONTH_ALIASES.items():
        if s_up.startswith(k):
            return v
    return None

def classify_equipment_type(v: str) -> str:
    """Clasifica BDI en Practicaja / Dispenser / Other por nombre o prefijo (H22/H34/H68)."""
    s = str(v).strip().upper()
    if "PRACTICAJA" in s or s.startswith("H34") or s.startswith("H68"):
        return "Practicaja"
    if "DISPENSER" in s or s.startswith("H22"):
        return "Dispenser"
    return "Other"
