# lib/ui.py
from datetime import datetime
import base64, os
import streamlit as st

def _img_b64(path: str) -> str|None:
    if not path or not os.path.exists(path): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def version_card(version: str, last_update: str):
    st.sidebar.markdown("""
    <style>
      .ver-card{
        border:1px solid #e6eaf1; border-radius:12px; padding:10px 12px;
        background:#f8fafc; margin:8px 0 14px 0;
      }
      .ver-card h4{margin:0 0 6px 0; font-size:13px; color:#0f172a;}
      .ver-card .item{font-size:12px; color:#334155; margin:2px 0;}
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="ver-card">
          <h4>📌 Dashboard Info</h4>
          <div class="item"><b>Version:</b> {version}</div>
          <div class="item"><b>Last update:</b> {last_update}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def brand_header(title: str, logo_path: str|None):
    logo_b64 = _img_b64(logo_path) if logo_path else None
    st.markdown("""
    <style>
      .brand-header{
        background:#F6F8FB; border:1px solid #e6eaf1;
        border-radius:14px; padding:12px 16px; margin:6px 0 10px 0;
      }
      .brand-title{font-size:20px; line-height:1.15; margin:0; color:#0F172A;}
      .brand-logo{display:flex; align-items:center; justify-content:center}
      .brand-logo img{max-height:90px}
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns([2, 8])
        with c1:
            if logo_b64:
                st.markdown(f'<div class="brand-logo"><img src="data:image/png;base64,{logo_b64}"/></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<h1 class="brand-title">{title}</h1>', unsafe_allow_html=True)

def brand_footer(brand_name: str, author_line: str, logo_path: str|None):
    logo_b64 = _img_b64(logo_path) if logo_path else None
    year = datetime.now().year
    st.markdown(f"""
    <style>
      .app-footer {{
        position: fixed; left: 0; right: 0; bottom: 0;
        background: #FFFFFF; border-top: 1px solid #e6eaf1;
        padding: 8px 14px; z-index: 9999;
        font-size: 13px; color:#475569;
      }}
      .app-footer .inner {{
        max-width: 1200px; margin: 0 auto;
        display:flex; gap:10px; align-items:center;
      }}
      .app-footer img {{ height:18px; }}
      .block-container{{ padding-bottom: 54px; }}
    </style>
    <div class="app-footer">
      <div class="inner">
        {f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else ''}
        <span>© {year} {brand_name} • {author_line}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
