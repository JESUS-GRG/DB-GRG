# =========================
# app_step4_bdi_state_idc.py
# Overview + Operations + BDI/State/IDC + Data Explorer (exports)
# =========================
# Requirements:
#   pip install streamlit plotly pyarrow pandas numpy openpyxl

import os
import io
import json
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# -------------------------
# Theme (centralized colors)
# -------------------------
THEME = {
    "years_palette": px.colors.sequential.Blues,
    "tickets": "#08306B",
    "bdi_prac": "#2272B5",
    "bdi_disp": "#08306B",
    "highlight": "#C81D25",
    "neutral": "#B0B0B0",
    "annotation_text": "#0F172A",
    "heatmap_bg": "#F5F7FA",  # very light background for number-only heatmap
}

# -------------------------
# Import project utilities (with robust fallbacks)
# -------------------------
try:
    from lib.utils import (
        MONTH_LABELS, LABEL2NUM, MONTH_ORDER, MONTH_ORDER_CAP, DAY_ORDER,
        norm_series_noaccents, apply_state_map, pct, fmt_pct, safe_default, year_palette
    )
except Exception:
    MONTH_ORDER = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT":10, "NOV":11, "DEC":12
    }
    MONTH_LABELS = {i: lbl.title() for i, lbl in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
    MONTH_ORDER_CAP = list(MONTH_LABELS.values())
    LABEL2NUM = {v: k for k, v in MONTH_LABELS.items()}

    def year_palette(years):
        yrs = sorted([str(y) for y in years])
        pal = THEME["years_palette"]
        cmap = {y: pal[i % len(pal)] for i, y in enumerate(yrs)}
        return yrs, cmap

    def apply_state_map(s):
        return pd.Series(s, dtype="object").astype(str).str.strip()

    def pct(a, b):
        return 0 if (b in [0, None] or pd.isna(b) or b == 0) else (100.0 * a / b)

    def fmt_pct(x):
        return f"{x:.1f}%"

    def safe_default(v, default):
        return default if v is None else v

try:
    from lib.data_io import load_curated, load_bdi, ensure_geojson, load_geojson
except Exception:
    def load_curated(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Curated parquet not found: {path}")
        return pd.read_parquet(path, engine="pyarrow")

    def load_bdi(path):
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path, engine="pyarrow")

    def ensure_geojson(primary, alt):
        return primary if os.path.exists(primary) else alt

    def load_geojson(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

try:
    from lib.ui import version_card, brand_header, brand_footer
except Exception:
    def version_card(version, last_update):
        st.sidebar.info(f"Version: {version}\n\nLast update: {last_update}")

    def brand_header(title, logo_path=None):
        cols = st.columns([1, 8])
        with cols[0]:
            if logo_path and os.path.exists(logo_path):
                st.image(logo_path, use_container_width=True)
        with cols[1]:
            st.title(title)

    def brand_footer(brand_name, author_line, logo_path=None):
        st.markdown("---")
        st.caption(f"{brand_name} — {author_line}")

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Operational Dashboard", layout="wide")

# Paths (adjust if needed)
base_path = r"C:\Users\jlimo\OneDrive\Documentos\GRG_Acumulado"
path_data = os.path.join(base_path, "Datos") if os.path.exists(os.path.join(base_path, "Datos")) else base_path
parquet_dir = os.path.join(path_data, "parquet")
curated_dir = os.path.join(parquet_dir, "curated")
curated_path = os.path.join(curated_dir, "df_resultado.parquet")
raw_bdi_path = os.path.join(parquet_dir, "raw_bdi.parquet")
raw_hc_path  = os.path.join(parquet_dir, "raw_hc.parquet")
geo_dir = os.path.join(path_data, "geo"); os.makedirs(geo_dir, exist_ok=True)
PRIMARY_GEO = os.path.join(geo_dir, "mx_states.geojson")
ALT_GEO     = os.path.join(geo_dir, "mexicoHigh.json")

# Branding/version
BRAND_DIR   = os.path.join(path_data, "branding"); os.makedirs(BRAND_DIR, exist_ok=True)
LOGO_PATH   = os.path.join(BRAND_DIR, "logo.png")
BRAND_NAME  = "GRG Banking"
DASH_TITLE  = "Operational Dashboard"
AUTHOR_LINE = "Prepared by: José de Jesús Sánchez Limón"
VERSION     = "4.0.0"
LAST_UPDATE = "Oct 2025"

# -------------------------
# Cached loaders
# -------------------------
@st.cache_data(show_spinner=True)
def _load_curated_cached(p):
    return load_curated(p)

@st.cache_data(show_spinner=True)
def _load_bdi_cached(p):
    return load_bdi(p)

@st.cache_data(show_spinner=False)
def _load_geo_cached(primary, alt):
    path = ensure_geojson(primary, alt)
    try:
        return load_geojson(path)
    except Exception:
        return {}

@st.cache_data(show_spinner=True)
def _load_hc_cached(p):
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")

# Load data
try:
    df = _load_curated_cached(curated_path)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

df_bdi = _load_bdi_cached(raw_bdi_path)
df_hc  = _load_hc_cached(raw_hc_path)
mx_geojson = _load_geo_cached(PRIMARY_GEO, ALT_GEO)

# =========================
# Helpers
# =========================
def _clean_month_num_column(data: pd.DataFrame) -> pd.DataFrame:
    if "_mnum" not in data.columns:
        if "Month" in data.columns:
            data = data.copy()
            mnum = pd.to_numeric(data["Month"], errors="coerce")
            if mnum.isna().all():
                tmp = data["Month"].astype(str).str.strip().str.upper()
                data["_mnum"] = tmp.map(MONTH_ORDER)
            else:
                data["_mnum"] = mnum
        else:
            data = data.copy()
            data["_mnum"] = np.nan
    return data

def _month_labels_from_nums(nums):
    return [MONTH_LABELS.get(int(n), str(int(n))) for n in nums if pd.notna(n)]

def _fmt_list(v):
    return ", ".join(map(str, v)) if v else "All"

def _build_filename(prefix: str, customer, years, months, region, state, ext: str):
    def _tokenize(x, key):
        if not x:
            return None
        if isinstance(x, (list, tuple, set)):
            val = "-".join([str(i) for i in x][:4])
        else:
            val = str(x)
        return f"{key}-{val}"

    parts = [prefix]
    parts.append(_tokenize(customer, "cust"))
    parts.append(_tokenize(years, "yr"))
    parts.append(_tokenize(months, "mo"))
    parts.append(_tokenize(region, "reg"))
    parts.append(_tokenize(state, "st"))
    parts = [p for p in parts if p]
    base = "_".join(parts)
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    return f"{base}.{ext}"

def _excel_with_filters_sheet(df_data: pd.DataFrame, filters: dict, sheet_name="data"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_data.to_excel(writer, index=False, sheet_name=sheet_name)
        meta = pd.DataFrame({"Filter": list(filters.keys())})
        meta["Value"] = [
            "" if v is None else (", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v))
            for v in filters.values()
        ]
        meta.to_excel(writer, index=False, sheet_name="Filters")
    output.seek(0)
    return output

def _compute_bdi_pct_change(df_bdi_raw: pd.DataFrame) -> tuple:
    """% change of total BDI between most recent month and previous (only BBVA; global)."""
    if df_bdi_raw is None or df_bdi_raw.empty:
        return (None, None, None, None, None)

    bdi = df_bdi_raw.copy()
    if "Customer" in bdi.columns:
        bdi = bdi[bdi["Customer"].astype(str).str.upper() == "BBVA"]
    if bdi.empty:
        return (None, None, None, None, None)

    bdi = _clean_month_num_column(bdi)
    bdi["Year"] = pd.to_numeric(bdi["Year"], errors="coerce")
    bdi = bdi.dropna(subset=["Year", "_mnum"]).copy()
    if bdi.empty: return (None, None, None, None, None)

    bdi["Date"] = pd.to_datetime(
        bdi["Year"].astype(int).astype(str) + "-" + bdi["_mnum"].astype(int).astype(str) + "-01",
        format="%Y-%m-%d", errors="coerce"
    )
    bdi = bdi.dropna(subset=["Date"])
    if bdi.empty: return (None, None, None, None, None)

    monthly = (bdi.groupby("Date", as_index=False).size().rename(columns={"size":"Count"}).sort_values("Date"))
    if monthly.shape[0] < 2: return (None, None, None, None, None)

    current_row = monthly.iloc[-1]; prev_row = monthly.iloc[-2]
    cur_cnt = int(current_row["Count"]); prev_cnt = int(prev_row["Count"])
    if prev_cnt == 0:
        return (None, current_row["Date"], prev_row["Date"], cur_cnt, prev_cnt)
    change = 100.0 * (cur_cnt - prev_cnt) / prev_cnt
    return (change, current_row["Date"], prev_row["Date"], cur_cnt, prev_cnt)

def _first_col(df_, candidates):
    for c in candidates:
        if c in df_.columns:
            return c
    return None

def _class2_equipment(v: str) -> str:
    s = str(v).strip().upper()
    if "PRACTICAJA" in s or s.startswith("H34") or s.startswith("H68"):
        return "Practicaja"
    if "DISPENSER" in s or s.startswith("H22"):
        return "Dispenser"
    return "Other"

def _norm_noacc(s: str) -> str:
    x = str(s)
    x = ''.join(ch for ch in unicodedata.normalize('NFD', x) if unicodedata.category(ch) != 'Mn')
    x = x.strip().lower().replace('_', ' ').replace('-', ' ')
    return ' '.join(x.split())

# =========================
# Header + Global Filters (NO download here)
# =========================
brand_header(DASH_TITLE, LOGO_PATH)

st.sidebar.markdown("## 🎛️ Global Filters")
version_card(VERSION, LAST_UPDATE)

# Ensure numeric month in tickets
df = _clean_month_num_column(df)

# Customer (single-select, default BBVA if available)
customers = sorted(df["Customer"].dropna().astype(str).unique()) if "Customer" in df.columns else []
default_idx = customers.index("BBVA") if ("BBVA" in customers) else 0 if customers else 0
customer_sel = st.sidebar.selectbox("Customer", customers, index=default_idx if customers else 0, key="g_customer")

# Years and dependent Months (Tickets only)
years_all = sorted([int(y) for y in df["Year"].dropna().unique()]) if "Year" in df.columns else []
years_sel = st.sidebar.multiselect("Years", years_all, default=years_all, key="g_years")

sub_df = df[df["Year"].isin(years_sel)] if years_sel else df
months_present = sorted([int(m) for m in sub_df["_mnum"].dropna().unique()]) if "_mnum" in sub_df.columns else []
month_options_labels = _month_labels_from_nums(months_present) if months_present else list(MONTH_LABELS.values())
months_sel_labels = st.sidebar.multiselect("Months (dependent on Years)", month_options_labels, default=month_options_labels, key="g_months")
months_sel_nums = [LABEL2NUM.get(lbl, None) for lbl in months_sel_labels]
months_sel_nums = [m for m in months_sel_nums if m is not None]

# Other global filters (TICKETS)
regions = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else []
region_sel = st.sidebar.multiselect("Region", regions, default=regions, key="g_region")

if "State" in df.columns:
    states_pool = df[df["Region"].isin(region_sel)]["State"].dropna().unique() if region_sel else df["State"].dropna().unique()
    states = sorted(states_pool.tolist())
else:
    states = []
state_sel = st.sidebar.multiselect("State", states, default=states, key="g_state")

assignments = sorted(df["Assignment"].dropna().unique()) if "Assignment" in df.columns else []
assignment_sel = st.sidebar.multiselect("Assignment", assignments, default=assignments, key="g_assignment")

sv_types = sorted(df["Service Type"].dropna().unique()) if "Service Type" in df.columns else []
stype_sel = st.sidebar.multiselect("Service Type", sv_types, default=sv_types, key="g_stype")

eq_types_all = sorted(df["Equipment Type"].dropna().unique()) if "Equipment Type" in df.columns else []
etype_sel = st.sidebar.multiselect("Equipment Type", eq_types_all, default=eq_types_all, key="g_etype")

status_vals = sorted(df["Status"].astype(str).str.upper().unique()) if "Status" in df.columns else []
default_status = ["SUCCESS"] if "SUCCESS" in status_vals else status_vals
status_sel = st.sidebar.multiselect("Status", status_vals, default=default_status, key="g_status")

reinc_vals_master = ["NO REINCIDENTE", "REINCIDENTE DND", "REINCIDENTE SE"]
reinc_in_df = df["Reincidet"].dropna().astype(str).unique().tolist() if "Reincidet" in df.columns else []
reinc_present = [v for v in reinc_vals_master if v in reinc_in_df] or reinc_vals_master
reinc_sel = st.sidebar.multiselect("Reincidences", reinc_present, default=reinc_present, key="g_reinc")

spare_options = ["YES", "NO", "BLANK"]
spare_sel = st.sidebar.multiselect("Spare Part Used", spare_options, default=spare_options, key="g_spare")

spr_options = []
if "SPR" in df.columns:
    spr_detected = [str(v) for v in df["SPR"].dropna().unique()]
    spr_options = [v for v in ["YES", "NO"] if v in spr_detected] + [v for v in sorted(set(spr_detected)) if v not in ["YES", "NO"]]
spr_sel = st.sidebar.multiselect("SPR", spr_options, default=spr_options, key="g_spr") if spr_options else []

# Build global mask for TICKETS (df)
gmask = pd.Series(True, index=df.index)

if customer_sel and "Customer" in df.columns:
    gmask &= df["Customer"].astype(str) == str(customer_sel)
if years_sel and "Year" in df.columns:
    gmask &= df["Year"].isin(years_sel)
if months_sel_nums:
    if "_mnum" in df.columns:
        gmask &= df["_mnum"].isin(months_sel_nums)
    elif "Month" in df.columns:
        gmask &= pd.to_numeric(df["Month"], errors="coerce").isin(months_sel_nums)
if status_sel and "Status" in df.columns:
    gmask &= df["Status"].astype(str).str.upper().isin([s.upper() for s in status_sel])
if "Reincidet" in df.columns and reinc_sel:
    gmask &= df["Reincidet"].isin(reinc_sel)
if spare_sel and "SparePartFlag" in df.columns:
    gmask &= df["SparePartFlag"].isin(spare_sel)
if stype_sel and "Service Type" in df.columns:
    gmask &= df["Service Type"].isin(stype_sel)
if etype_sel and "Equipment Type" in df.columns:
    gmask &= df["Equipment Type"].isin(etype_sel)
if region_sel and "Region" in df.columns:
    gmask &= df["Region"].isin(region_sel)
if state_sel and "State" in df.columns:
    gmask &= df["State"].isin(state_sel)
if spr_sel and "SPR" in df.columns:
    gmask &= df["SPR"].astype(str).isin(spr_sel)
if assignment_sel and "Assignment" in df.columns:
    gmask &= df["Assignment"].isin(assignment_sel)

df_global = df.loc[gmask].copy()

# Chips (compact summary)
st.markdown("""
<style>
.chips {display:flex; flex-wrap:wrap; gap:.4rem; margin:.25rem 0 .5rem 0;}
.chip {background:#eef2f7; color:#0F172A; padding:.15rem .55rem; border-radius:999px; font-size:.85rem;}
.chip b{color:#023e8a;}
</style>
""", unsafe_allow_html=True)

chips_html = f"""
<div class="chips">
  <span class="chip"><b>Customer:</b> {customer_sel or "All"}</span>
  <span class="chip"><b>Region:</b> {_fmt_list(region_sel)}</span>
  <span class="chip"><b>State:</b> {_fmt_list(state_sel)}</span>
  <span class="chip"><b>Service:</b> {_fmt_list(stype_sel)}</span>
  <span class="chip"><b>Years:</b> {_fmt_list(years_sel)}</span>
  <span class="chip"><b>Months:</b> {_fmt_list(months_sel_labels)}</span>
</div>
"""
st.markdown(chips_html, unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Overview",
    "Operations",
    "BDI / State / IDC",
    "Data Explorer"
])

# =========================
# OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Overview")

    if df_global.empty:
        st.info("No data for current filters.")
    else:
        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)

        total_tk = int(len(df_global))
        succ_cnt = (df_global["Status"].astype(str).str.upper() == "SUCCESS").sum() if "Status" in df_global.columns else 0
        succ_pct = pct(succ_cnt, total_tk)

        rec_cnt = (df_global["Reincidet"] == "REINCIDENTE SE").sum() if "Reincidet" in df_global.columns else 0
        rec_pct = pct(rec_cnt, total_tk)

        sp_yes_cnt = (df_global["SparePartFlag"] == "YES").sum() if "SparePartFlag" in df_global.columns else 0
        sp_yes_pct = pct(sp_yes_cnt, total_tk)

        c1.metric("Tickets (Filtered)", f"{total_tk:,}")
        c2.metric("% Success", f"{succ_pct:.1f}%", help="Share of SUCCESS over filtered tickets")
        c3.metric("% Reincidences", f"{rec_pct:.1f}%", help="Share of REINCIDENTE SE over filtered tickets")
        c4.metric("% Spare YES", f"{sp_yes_pct:.1f}%", help="Share of SparePartFlag == YES over filtered tickets")

        # % BDI KPI — only Customer=BBVA; independent of global filters
        if str(customer_sel).upper() == "BBVA":
            change, cur_date, prev_date, cur_cnt, prev_cnt = _compute_bdi_pct_change(df_bdi)
            if change is None:
                c5.metric("% BDI (MoM)", "N/A", help="Month-over-month change of total BDI (independent from filters)")
            else:
                cur_label = cur_date.strftime("%b %Y"); prev_label = prev_date.strftime("%b %Y")
                delta_str = f"{change:+.1f}% vs {prev_label}"
                c5.metric("% BDI (MoM)", f"{change:.1f}%", delta=delta_str,
                          help=f"MoM change of total BDI (independent): {prev_label}={prev_cnt:,} → {cur_label}={cur_cnt:,}")

        st.markdown("---")

        # Monthly Trend (YoY)
        if {"Year", "_mnum"}.issubset(df_global.columns):
            tr = (
                df_global.dropna(subset=["Year", "_mnum"])
                         .groupby(["Year", "_mnum"], as_index=False)
                         .size().rename(columns={"size": "Count"})
            )
            if not tr.empty:
                tr["YearStr"] = tr["Year"].astype(int).astype(str)
                tr["MonthLabel"] = tr["_mnum"].map(MONTH_LABELS)
                years_order, cmap = year_palette(tr["YearStr"].unique())

                fig_tr = px.line(
                    tr, x="MonthLabel", y="Count", color="YearStr",
                    category_orders={"MonthLabel": MONTH_ORDER_CAP, "YearStr": years_order},
                    color_discrete_map=cmap
                )
                fig_tr.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=6))
                fig_tr.update_layout(
                    title="Monthly Trend (Year-over-Year)",
                    xaxis_title="Month", yaxis_title="Tickets",
                    legend_title_text="Year",
                    height=420, margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_tr, use_container_width=True)
            else:
                st.info("Monthly trend: no data for current filters.")
        else:
            st.info("Monthly trend: missing 'Year' or month column in dataset.")

        st.markdown("---")

        # Status Mix (Counts/%)
        if {"Year", "Status"}.issubset(df_global.columns):
            base = df_global.copy()
            base["YearStr"] = base["Year"].astype("Int64").astype(str)
            base["StatusUp"] = base["Status"].astype(str).str.upper()
            agg = (
                base.dropna(subset=["YearStr", "StatusUp"])
                    .groupby(["YearStr", "StatusUp"], as_index=False)
                    .size().rename(columns={"size": "Count"})
            )

            if not agg.empty:
                mode = st.radio("Display Mode", ["Counts", "Percentages"], index=0, horizontal=True, key="ovw_status_mode")
                years_order, cmap = year_palette(agg["YearStr"].unique())
                plot_df = agg.copy()
                if mode == "Percentages":
                    plot_df["YearTotal"] = plot_df.groupby("YearStr")["Count"].transform("sum")
                    plot_df["Pct"] = 100.0 * plot_df["Count"] / plot_df["YearTotal"]
                    y_col = "Pct"; yaxis_kwargs = dict(yaxis_title="Share (%)", yaxis=dict(ticksuffix="%")); text_tpl = "%{y:.0f}%"
                else:
                    y_col = "Count"; yaxis_kwargs = dict(yaxis_title="Tickets"); text_tpl = "%{y}"

                order_status = plot_df.groupby("StatusUp")[y_col].sum().sort_values(ascending=False).index.tolist()
                fig_st = px.bar(
                    plot_df, x="YearStr", y=y_col, color="StatusUp",
                    barmode="stack",
                    category_orders={"YearStr": years_order, "StatusUp": order_status},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_st.update_traces(texttemplate=text_tpl, textposition="outside", cliponaxis=False,
                                     marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1)
                fig_st.update_layout(
                    title=f"Status Mix by Year — {mode}",
                    xaxis_title="Year", legend_title_text="Status",
                    height=420, bargap=0.18, margin=dict(t=40, b=10, l=10, r=10), **yaxis_kwargs
                )
                st.plotly_chart(fig_st, use_container_width=True)
            else:
                st.info("Status mix: no data for current filters.")
        else:
            st.info("Status mix: missing 'Year' or 'Status' in dataset.")

# =========================
# OPERATIONS
# =========================
with tabs[1]:
    st.subheader("Operations")

    if df_global.empty:
        st.info("No data for current filters.")
    else:
        # Service Mix
        st.markdown("### Service Mix — Top-N")
        if {"Year", "Service Type"}.issubset(df_global.columns):
            years_ops = sorted([int(y) for y in df_global["Year"].dropna().unique()])
            if years_ops:
                y_idx = len(years_ops) - 1
                y_selected = st.selectbox("Year", years_ops, index=y_idx, key="ops_mix_year")
                topn = st.slider("Top-N services", min_value=3, max_value=25, value=10, step=1, key="ops_mix_topn")
                df_y = df_global[df_global["Year"] == y_selected].copy()
                svc_counts = (
                    df_y["Service Type"].astype(str).str.strip()
                        .value_counts(dropna=False).rename_axis("Service Type")
                        .reset_index(name="Count").sort_values("Count", ascending=True)
                )
                if svc_counts.empty:
                    st.info("No service data for the selected year.")
                else:
                    plot_df = svc_counts.tail(topn)
                    fig_mix = px.bar(plot_df, x="Count", y="Service Type", orientation="h", text="Count")
                    fig_mix.update_traces(textposition="outside", cliponaxis=False,
                                          marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1)
                    fig_mix.update_layout(
                        title=f"Top {topn} Services — {y_selected}",
                        xaxis_title="Tickets", yaxis_title=None,
                        height=520, bargap=0.18, margin=dict(t=40, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_mix, use_container_width=True)
            else:
                st.info("Service mix: no valid years detected.")
        else:
            st.info("Service mix: missing 'Year' or 'Service Type' in dataset.")

        st.markdown("---")

        # By Region
        st.markdown("### Performance by Region — Tickets per Year")
        if {"Region", "Year"}.issubset(df_global.columns):
            reg = df_global.groupby(["Region", "Year"], as_index=False).size().rename(columns={"size":"Count"})
            if reg.empty:
                st.info("No region data for current filters.")
            else:
                reg["YearStr"] = reg["Year"].astype(int).astype(str)
                years_order, cmap = year_palette(reg["YearStr"].unique())
                order_regions = reg.groupby("Region")["Count"].sum().sort_values(ascending=False).index.tolist()
                fig_reg = px.bar(
                    reg, x="Region", y="Count", color="YearStr", barmode="group",
                    category_orders={"Region": order_regions, "YearStr": years_order},
                    color_discrete_map=cmap, text="Count"
                )
                fig_reg.update_traces(textposition="outside", textangle=-90, cliponaxis=False,
                                      marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1)
                fig_reg.update_layout(
                    title="Tickets by Region and Year",
                    xaxis_title=None, yaxis_title="Tickets", legend_title_text="Year",
                    height=520, bargap=0.18, margin=dict(t=35, b=10, l=10, r=10)
                )
                fig_reg.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("By Region: missing 'Region' or 'Year' in dataset.")

        st.markdown("---")

        # By State
        st.markdown("### Performance by State — Tickets per Year")
        if {"State", "Year"}.issubset(df_global.columns):
            sta = df_global.groupby(["State", "Year"], as_index=False).size().rename(columns={"size":"Count"})
            if sta.empty:
                st.info("No state data for current filters.")
            else:
                sta["YearStr"] = sta["Year"].astype(int).astype(str)
                years_order, cmap = year_palette(sta["YearStr"].unique())
                order_states = sta.groupby("State")["Count"].sum().sort_values(ascending=False).index.tolist()
                fig_sta = px.bar(
                    sta, x="State", y="Count", color="YearStr", barmode="group",
                    category_orders={"State": order_states, "YearStr": years_order},
                    color_discrete_map=cmap, text="Count"
                )
                fig_sta.update_traces(textposition="outside", textangle=-90, cliponaxis=False,
                                      marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1)
                fig_sta.update_layout(
                    title="Tickets by State and Year",
                    xaxis_title=None, yaxis_title="Tickets", legend_title_text="Year",
                    height=520, bargap=0.18, margin=dict(t=35, b=10, l=10, r=10)
                )
                fig_sta.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_sta, use_container_width=True)
        else:
            st.info("By State: missing 'State' or 'Year' in dataset.")

        st.markdown("---")

        # Heatmap — Month × Weekday (number-only, no palette)
        st.markdown("### Heatmap — Month × Weekday (Counts)")
        if "Diasem" in df_global.columns and ("_mnum" in df_global.columns or "Month" in df_global.columns):
            hm = df_global.copy()
            if "_mnum" not in hm.columns:
                hm["_mnum"] = pd.to_numeric(hm["Month"], errors="coerce")
            hm = (hm.dropna(subset=["Diasem", "_mnum"])
                    .groupby(["Diasem", "_mnum"], as_index=False)
                    .size().rename(columns={"size":"Count"}))
            if not hm.empty:
                hm["MonthLabel"] = hm["_mnum"].map(MONTH_LABELS)
                days_order = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
                mat = hm.pivot_table(index="Diasem", columns="MonthLabel", values="Count", aggfunc="sum", fill_value=0)
                mat = mat.reindex(index=[d for d in days_order if d in mat.index])
                mat = mat.reindex(columns=[m for m in MONTH_ORDER_CAP if m in mat.columns])
                z = mat.values.astype(int)
                x = list(mat.columns)
                y = list(mat.index)

                fig_hm = go.Figure(data=go.Heatmap(
                    z=[[0 for _ in x] for __ in y],
                    x=x, y=y,
                    colorscale=[[0, THEME["heatmap_bg"]],[1, THEME["heatmap_bg"]]],
                    showscale=False,
                    hoverinfo="skip"
                ))
                for i, yi in enumerate(y):
                    for j, xj in enumerate(x):
                        val = int(z[i, j]) if isinstance(z[i, j], (int, np.integer)) else int(round(z[i, j]))
                        fig_hm.add_annotation(
                            x=xj, y=yi, text=f"{val:,}",
                            showarrow=False, font=dict(size=12, color=THEME["annotation_text"])
                        )
                fig_hm.update_xaxes(title_text="Month")
                fig_hm.update_yaxes(title_text=None)
                fig_hm.update_layout(
                    title="Tickets Heatmap by Month and Weekday (Number-only)",
                    height=420, margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Heatmap: no Month × Weekday data for current filters.")
        else:
            st.info("Heatmap: missing 'Diasem' and month information in dataset.")

# =========================
# BDI / State / IDC  (overlay + IDC strip)
# =========================
with tabs[2]:
    st.subheader("BDI / State / IDC")

    if str(customer_sel).upper() != "BBVA":
        st.info("BDI views are available only for Customer = BBVA.")
        st.stop()

    if df_bdi.empty:
        st.info("BDI dataset is empty or not found.")
        st.stop()

    # ---------- helpers ----------
    def _first_col(df_, candidates):
        for c in candidates:
            if c in df_.columns: return c
        return None

    def _class2_equipment(v: str) -> str:
        s = str(v).strip().upper()
        if "PRACTICAJA" in s or s.startswith("H34") or s.startswith("H68"): return "Practicaja"
        if "DISPENSER" in s or s.startswith("H22"):                     return "Dispenser"
        return "Other"

    # ---- función EXACTA de conteo de IDC por estado (como la versión buena)
    def _hc_rows_by_state_count(df_hc_src: pd.DataFrame,
                                assignments_keep: list | None,
                                state_clean_fn=apply_state_map) -> pd.DataFrame:
        """
        - Excluye blacklist fija (workshop, regional assistant/head, chief idc)
        - Si 'assignments_keep' viene, además filtra a SOLO esos assignments (después de excluir).
        - Devuelve filas por StateClean: columnas -> ['StateClean','IDC_Count']
        """
        hc_state_counts = pd.DataFrame(columns=["StateClean", "IDC_Count"])
        try:
            if df_hc_src is not None and not df_hc_src.empty:
                def _first_col_local(df_, cands):
                    for c in cands:
                        if c in df_.columns:
                            return c
                    return None

                COL_HC_STATE  = _first_col_local(df_hc_src, ["State", "Estado"])
                COL_HC_ASSIGN = _first_col_local(df_hc_src, ["Assignment", "Asignación", "Asignacion"])

                if COL_HC_STATE:
                    hc_tmp = df_hc_src.copy()

                    if COL_HC_ASSIGN:
                        EXCLUDE_ASSIGN = {"workshop", "regional assistant", "regional head", "chief idc"}
                        hc_tmp = hc_tmp[
                            ~hc_tmp[COL_HC_ASSIGN].astype(str).str.strip().str.lower().isin(EXCLUDE_ASSIGN)
                        ]
                        if assignments_keep:
                            keep_norm = { _norm_noacc(x) for x in assignments_keep }
                            hc_tmp = hc_tmp[
                                hc_tmp[COL_HC_ASSIGN].astype(str).map(_norm_noacc).isin(keep_norm)
                            ]

                    hc_tmp["StateClean"] = state_clean_fn(hc_tmp[COL_HC_STATE])
                    hc_state_counts = (
                        hc_tmp["StateClean"]
                        .value_counts(dropna=False)
                        .rename_axis("StateClean")
                        .reset_index(name="IDC_Count")
                    )
        except Exception:
            hc_state_counts = pd.DataFrame(columns=["StateClean", "IDC_Count"])

        return hc_state_counts

    # ---------- BDI filtros locales ----------
    bdi_base = df_bdi.copy()
    reg_col = _first_col(bdi_base, ["Region", "REGION", "region"])
    if reg_col:
        bdi_base["RegionStd"] = (bdi_base[reg_col].astype("string")
                                 .str.strip()
                                 .replace({"Metro Norte":"Metro", "Metro Sur":"Metro"}))
    else:
        bdi_base["RegionStd"] = None

    bdi_base = _clean_month_num_column(bdi_base)
    bdi_base["Year"] = pd.to_numeric(bdi_base["Year"], errors="coerce")

    with st.expander("BDI Local Filters", expanded=True):
        regs_all = sorted(bdi_base["RegionStd"].dropna().unique()) if "RegionStd" in bdi_base.columns else []
        regs_sel = st.multiselect("Region (BDI)", regs_all, default=regs_all, key="bdi_regs")

        years_bdi = sorted([int(y) for y in bdi_base["Year"].dropna().unique()])
        y_def = max(years_bdi) if years_bdi else None
        y_sel = st.selectbox("BDI Year", years_bdi,
                             index=(years_bdi.index(y_def) if y_def in years_bdi else 0) if years_bdi else 0,
                             key="bdi_year")

        m_tbl = (bdi_base[bdi_base["Year"] == y_sel][["_mnum"]]
                 .dropna().drop_duplicates().sort_values("_mnum"))
        if not m_tbl.empty:
            mnums = m_tbl["_mnum"].astype(int).tolist()
            labels = [MONTH_LABELS[m] for m in mnums]
            m_label_sel = st.selectbox("BDI Month", labels, index=len(labels)-1, key="bdi_month")
            mnum_sel = mnums[labels.index(m_label_sel)]
        else:
            st.info("No months available for the selected BDI year.")
            mnum_sel = None

        types_all = ["Practicaja", "Dispenser"]
        types_sel = st.multiselect("Equipment classes (for KPIs and bars by State)",
                                   types_all, default=types_all, key="bdi_types")

        # ---- Filtro LOCAL (IDC / HC) por Assignment, con defaults pedidos
        idc_assign_keep = None
        if not df_hc.empty:
            _hc_assign_col = _first_col(df_hc, ["Assignment", "Asignación", "Asignacion"])
            if _hc_assign_col:
                hc_vals = (
                    df_hc[_hc_assign_col]
                    .dropna().astype(str).str.strip()
                    .sort_values(key=lambda s: s.str.lower())
                    .unique().tolist()
                )
                default_ids = ["IDC.SE", "IDC.DND", "IDC.Impl", "IDC.VC"]
                defaults_present = [v for v in default_ids if v in hc_vals] or hc_vals
                idc_assign_keep = st.multiselect(
                    "IDC Assignment (local, para conteo IDC)",
                    hc_vals,
                    default=defaults_present,
                    help="Afecta SOLO el conteo de IDC por estado (franja inferior)."
                )

    # aplicar filtros locales (BDI)
    bdi_cut = bdi_base.copy()
    if regs_sel:   bdi_cut = bdi_cut[bdi_cut["RegionStd"].isin(regs_sel)]
    if y_sel is not None:   bdi_cut = bdi_cut[bdi_cut["Year"] == y_sel]
    if mnum_sel is not None: bdi_cut = bdi_cut[bdi_cut["_mnum"] == mnum_sel]

    eq_col = _first_col(bdi_cut, ["Equipment Type", "Equipment", "Type"])
    bdi_cut["Class2"] = bdi_cut[eq_col].map(_class2_equipment) if eq_col else "Other"
    bdi_f = bdi_cut[bdi_cut["Class2"].isin(types_sel)] if types_sel else bdi_cut

    # KPIs
    if not bdi_f.empty:
        total_eq = len(bdi_f)
        n_prac   = (bdi_f["Class2"] == "Practicaja").sum()
        n_disp   = (bdi_f["Class2"] == "Dispenser").sum()
        k1, k2, k3 = st.columns(3)
        k1.metric("BDI Inventory (selected cut)", f"{total_eq:,}", help=f"Year {y_sel}, Month {m_label_sel}")
        k2.metric("Practicaja", f"{n_prac:,}")
        k3.metric("Dispenser", f"{n_disp:,}")
    else:
        st.info("No BDI data for the selected local filters.")

    # ---------- BDI vs Tickets + IDC strip ----------
    st.markdown("### BDI vs Tickets by State (overlay)")

    # BDI barras apiladas por estado
    if "State" in bdi_f.columns:
        tmp = bdi_f.copy()
        tmp["StateClean"] = apply_state_map(tmp["State"])
        sta_stack = (tmp[tmp["Class2"].isin(["Practicaja", "Dispenser"])]
                       .groupby(["StateClean", "Class2"], as_index=False)
                       .size().rename(columns={"size":"Count"}))
    else:
        sta_stack = pd.DataFrame()

    # Tickets (del contexto global de tickets) — overlay 1 año
    tk_base = df_global.copy()
    if "State" in tk_base.columns:
        tk_base["StateClean"] = apply_state_map(tk_base["State"])
    tk_agg = (tk_base.dropna(subset=["StateClean", "Year"])
                    .groupby(["Year", "StateClean"], as_index=False)
                    .size().rename(columns={"size":"TK"})) if not tk_base.empty else pd.DataFrame()
    years_u = sorted(tk_agg["Year"].unique().tolist()) if not tk_agg.empty else []

    # IDC por estado — usando la función correcta (sin filtrar por región para evitar ceros)
    idc_counts = _hc_rows_by_state_count(
        df_hc_src=df_hc,
        assignments_keep=idc_assign_keep,
        state_clean_fn=apply_state_map
    )

    if sta_stack.empty or tk_agg.empty:
        st.info("Not enough data to build the combined view.")
    else:
        totals = sta_stack.groupby("StateClean")["Count"].sum().sort_values(ascending=False)
        order_states = totals.index.tolist()

        piv = (sta_stack.pivot_table(index="StateClean", columns="Class2", values="Count",
                                     aggfunc="sum", fill_value=0).reindex(order_states))
        y_prac = piv["Practicaja"].values if "Practicaja" in piv.columns else [0]*len(order_states)
        y_disp = piv["Dispenser"].values  if "Dispenser"  in piv.columns else [0]*len(order_states)

        y_options = sorted(years_u)
        y_sel_single = st.selectbox("Year for Tickets (overlay)", y_options,
                                    index=len(y_options)-1 if y_options else 0,
                                    key="bdi_overlay_year")
        ser_tk = (tk_agg[tk_agg["Year"] == y_sel_single]
                    .set_index("StateClean")["TK"].reindex(order_states)
                    .fillna(0).astype(float).tolist()) if y_options else [0]*len(order_states)

        # IDC strip alineado
        idc_aligned = (idc_counts.set_index("StateClean")
                                   .reindex(order_states)["IDC_Count"]
                                   .fillna(0).astype(int).tolist()) if not idc_counts.empty else [0]*len(order_states)
        idc_max = max(idc_aligned) if idc_aligned else 0

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.78, 0.22],
            specs=[[{"secondary_y": True}], [{"type": "heatmap"}]]
        )

        # Barras BDI
        fig.add_trace(go.Bar(
            name="Practicaja", x=order_states, y=y_prac, marker_color=THEME["bdi_prac"],
            text=[f"{int(v):,}" for v in y_prac], textposition="outside", textangle=-90,
            cliponaxis=False, textfont=dict(size=10, color="#000000"),
            marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1, opacity=0.9
        ), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(
            name="Dispenser", x=order_states, y=y_disp, marker_color=THEME["bdi_disp"],
            text=[f"{int(v):,}" for v in y_disp], textposition="outside", textangle=-90,
            cliponaxis=False, textfont=dict(size=10, color="#000000"),
            marker_line_color="rgba(0,0,0,0.25)", marker_line_width=1, opacity=0.9
        ), row=1, col=1, secondary_y=False)

        # Línea de tickets (overlay)
        fig.add_trace(go.Scatter(
            name=f"Tickets {int(y_sel_single)}", x=order_states, y=ser_tk,
            mode="lines+markers+text",
            text=[f"{int(v):,}" for v in ser_tk], textposition="top center", textfont=dict(size=9),
            line=dict(width=3, color=THEME["highlight"]), marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Tickets %{y:,}<extra></extra>"
        ), row=1, col=1, secondary_y=True)

        # Franja IDC (1 × N)
        fig.add_trace(go.Heatmap(
            z=[idc_aligned], x=order_states, y=["IDC"],
            colorscale=px.colors.sequential.Blues, zmin=0, zmax=(idc_max if idc_max>0 else 1),
            showscale=True, colorbar=dict(title="IDC", thickness=12, len=0.85, y=0.14, yanchor="bottom")
        ), row=2, col=1)
        for j, xj in enumerate(order_states):
            fig.add_annotation(
                x=xj, y="IDC", text=f"{int(idc_aligned[j]):,}",
                showarrow=False, font=dict(size=12, color=THEME["annotation_text"]),
                row=2, col=1
            )

        fig.update_layout(
            title="BDI by State (Stacked) with Tickets Overlay + IDC Density",
            barmode="stack", uniformtext_minsize=10, uniformtext_mode="hide",
            legend_title_text=None, height=740, bargap=0.18,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        fig.update_xaxes(tickangle=-35, row=1, col=1)
        fig.update_xaxes(tickangle=-35, row=2, col=1)
        fig.update_yaxes(title_text="BDI Equipment", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Tickets", row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text=None, row=2, col=1, showgrid=False)

        st.plotly_chart(fig, use_container_width=True)

    # ---------- BDI por Región — Treemaps separados ----------
    st.markdown("### BDI by Region — Treemap (separate by class)")
    if not bdi_f.empty and "RegionStd" in bdi_f.columns:
        # Practicaja
        prac_df = bdi_f[bdi_f["Class2"] == "Practicaja"]
        if not prac_df.empty:
            g_prac = (prac_df.groupby("RegionStd", as_index=False).size().rename(columns={"size":"Count"}))
            fig_tp = px.treemap(g_prac, path=["RegionStd"], values="Count",
                                title=f"BDI Treemap — Practicaja • Year {y_sel} • Month {m_label_sel}")
            fig_tp.update_traces(root_color="lightpink")
            fig_tp.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=420)
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("No Practicaja data for current cut.")

        # Dispenser
        disp_df = bdi_f[bdi_f["Class2"] == "Dispenser"]
        if not disp_df.empty:
            g_disp = (disp_df.groupby("RegionStd", as_index=False).size().rename(columns={"size":"Count"}))
            fig_td = px.treemap(g_disp, path=["RegionStd"], values="Count",
                                title=f"BDI Treemap — Dispenser • Year {y_sel} • Month {m_label_sel}")
            fig_td.update_traces(root_color="lightpink")
            fig_td.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=420)
            st.plotly_chart(fig_td, use_container_width=True)
        else:
            st.info("No Dispenser data for current cut.")
    else:
        st.info("Treemap: no BDI/Region data for current local filters.")

# =========================
# DATA EXPLORER — stable export (Tickets only + column picker)
# =========================
with tabs[3]:
    st.subheader("Data Explorer")

    st.info("Review and download Tickets with the current global filters. Export includes a Filters sheet (Excel) and a filename with key tokens.")

    # Vista previa opcional
    show_prev = st.checkbox("Show quick preview (first 200 rows)", value=False)
    if show_prev:
        st.dataframe(df_global.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⬇️ Export Tickets")

    # Selector de columnas
    all_cols = list(df_global.columns)
    cols_default = [c for c in all_cols if c.lower() in
                    {"customer","year","_mnum","month","region","state","idc","assignment","service type","status"}]
    cols_default = [c for c in cols_default if c in all_cols] or all_cols[: min(12, len(all_cols))]
    cols_pick = st.multiselect("Columns to include", all_cols, default=cols_default)

    export_format = st.radio("Format", ["Excel (.xlsx)", "CSV (.csv)"], index=0, horizontal=True, key="exp_fmt_tk")

    # Funciones de export
    def _filters_snapshot_dict():
        return {
            "Customer": customer_sel,
            "Years": years_sel,
            "Months": months_sel_labels,
            "Region": region_sel,
            "State": state_sel,
            "Service Type": stype_sel,
            "Equipment Type": etype_sel,
            "Status": status_sel,
            "Reincidences": reinc_sel,
            "Spare Part Used": spare_sel,
            "SPR": spr_sel,
            "Assignment": assignment_sel,
            "Exported at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _excel_with_filters(df_data: pd.DataFrame, filters: dict, sheet_name="tickets"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_data.to_excel(writer, index=False, sheet_name=sheet_name)
            meta = pd.DataFrame({"Filter": list(filters.keys())})
            meta["Value"] = [
                "" if v is None else (", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v))
                for v in filters.values()
            ]
            meta.to_excel(writer, index=False, sheet_name="Filters")
        output.seek(0)
        return output

    # Botón de descarga
    if st.button("Prepare file", type="primary", use_container_width=True, key="prep_tickets"):
        df_to_save = df_global[cols_pick].copy() if cols_pick else df_global.copy()
        if export_format.startswith("Excel"):
            file_bytes = _excel_with_filters(df_to_save, _filters_snapshot_dict(), sheet_name="tickets")
            fname = _build_filename("tickets", customer_sel, years_sel, months_sel_labels, region_sel, state_sel, "xlsx")
            st.download_button(
                label="Download Tickets (Excel)",
                data=file_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_tk_xlsx",
                use_container_width=True
            )
        else:
            csv_bytes = df_to_save.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            fname = _build_filename("tickets", customer_sel, years_sel, months_sel_labels, region_sel, state_sel, "csv")
            st.download_button(
                label="Download Tickets (CSV)",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
                key="dl_tk_csv",
                use_container_width=True
            )

# Footer
brand_footer(BRAND_NAME, AUTHOR_LINE, LOGO_PATH)
