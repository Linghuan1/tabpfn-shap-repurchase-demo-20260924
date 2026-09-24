from __future__ import annotations

import base64
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
import streamlit_shadcn_ui as ui
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, JsCode
from streamlit_option_menu import option_menu

TabPFNClassifier = None
if os.environ.get("ENABLE_TABPFN") == "1":
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        pass


ROOT = Path(__file__).resolve().parent

AUTH_CONFIG = {
    "credentials": {
        "usernames": {
            "admin": {
                "email": "admin@example.com",
                "name": "运营管理员",
                "password": "$2b$12$fyBl.8uhhIkpKBjY6fndgeCIs6V2SAsPt6UeMdaudyz3rGhZD1rpm",
            }
        }
    },
    "cookie": {
        "name": "prepared_food_repurchase_cookie",
        "expiry_days": 3,
    },
}

DATA_FILES = {
    "raw": ROOT / "江浙沪预制菜数据.csv",
    "preprocessed": ROOT / "1_预处理未编码数据集.csv",
    "encoded": ROOT / "3_预制菜建模数据集_已编码.csv",
    "mapping": ROOT / "2_特征编码映射表.csv",
    "metrics": ROOT / "结果表" / "模型性能汇总.csv",
    "coastal_metrics": ROOT / "结果表" / "东南沿海模型性能汇总.csv",
    "cat_balance": ROOT / "结果表" / "分类变量分布均衡性检验.csv",
    "num_balance": ROOT / "结果表" / "连续变量分布均衡性检验(多准则评估).csv",
}

IMAGE_FILES = {
    "target": ROOT / "图片" / "目标变量分布.png",
    "matrix": ROOT / "图片" / "矩阵风格混淆矩阵对比图.png",
    "dimension": ROOT / "图片" / "TabPFN_维度图.png",
    "beeswarm": ROOT / "图片" / "TabPFN_SHAP蜂巢图.png",
    "dependence": ROOT / "图片" / "SHAP依赖图_组合_1.png",
    "heatmap": ROOT / "图片" / "SHAP_全局热力图.png",
}

FEATURE_ORDER = [
    "单价(元)",
    "数量(个)",
    "消费城市级别",
    "支付方式",
    "优惠方式",
    "优惠金额(元)",
    "实际支付(元)",
    "库存剩余(件)",
    "商品评分",
    "客户年龄段",
    "商品类别",
    "包装规格(g)",
    "是否节假日",
    "是否用餐高峰",
]

ENCODERS = {
    "商品类别": {"主食面汤": 0, "蔬菜": 1, "豆制品": 2, "肉食": 3},
    "支付方式": {"银行卡": 0, "云闪付": 1, "微信": 2, "支付宝": 3, "余额": 4},
    "优惠方式": {"无": 0, "平台券": 1, "店铺券": 2, "满减": 3, "秒杀": 4, "会员折扣": 5},
    "客户年龄段": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4},
    "消费城市级别": {"一线": 0, "新一线": 1, "二线": 2, "三线": 3},
    "是否节假日": {"否": 0, "是": 1},
    "是否用餐高峰": {"否": 0, "是": 1},
}

APP_PAGES = ["总览", "复购预测", "数据与模型", "解释策略"]

PAGE_ICONS = {
    "总览": "speedometer2",
    "复购预测": "person-check",
    "数据与模型": "database-check",
    "解释策略": "diagram-3",
}


st.set_page_config(
    page_title="预制菜复购预测运营工作台",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fresh-bg: #F7FAF6;
            --surface: #FFFFFF;
            --surface-soft: #F1F7F0;
            --ink: #17211A;
            --muted: #66756D;
            --line: #DDE7DD;
            --green: #1F7A4D;
            --green-2: #18A058;
            --green-dark: #10251A;
            --amber: #F4A62A;
            --red: #E45745;
            --cyan: #2B8C9C;
            --pink: #E06C7F;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(247, 250, 246, .98) 0%, rgba(241, 247, 240, .98) 100%);
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        [data-testid="stSidebar"] {
            background: #0F2118;
            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] * {
            color: rgba(255,255,255,.88);
        }

        [data-testid="stSidebar"] .stButton > button {
            border-color: rgba(255,255,255,.22);
            color: #fff;
            background: rgba(255,255,255,.08);
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0 !important;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 16px 35px rgba(31, 122, 77, .07);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            border-radius: 8px !important;
            border: 1px solid #1F7A4D !important;
            background: #1F7A4D !important;
            color: white !important;
            font-weight: 700 !important;
            min-height: 42px;
            box-shadow: 0 10px 20px rgba(31, 122, 77, .18);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #16613D !important;
            border-color: #16613D !important;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #FFFFFF;
            box-shadow: 0 16px 40px rgba(23, 33, 26, .06);
            padding: 18px 18px 8px;
        }

        input, textarea, [data-baseweb="select"] {
            border-radius: 8px !important;
        }

        .app-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 18px 20px;
            background: rgba(255,255,255,.82);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(31, 122, 77, .08);
            margin-bottom: 18px;
        }

        .app-title h1 {
            margin: 0;
            font-size: 25px;
            line-height: 1.2;
            font-weight: 850;
        }

        .app-title p {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 14px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 11px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #F8FCF7;
            color: var(--muted);
            font-size: 13px;
            white-space: nowrap;
        }

        .section-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 16px 38px rgba(23, 33, 26, .055);
            margin-bottom: 16px;
        }

        .hero-band {
            position: relative;
            overflow: hidden;
            padding: 22px;
            border: 1px solid rgba(31,122,77,.18);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(31,122,77,.12), rgba(43,140,156,.07)),
                #FFFFFF;
            box-shadow: 0 18px 45px rgba(31,122,77,.08);
            margin-bottom: 16px;
        }

        .hero-band::after {
            content: "";
            position: absolute;
            right: -80px;
            top: -110px;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(244,166,42,.22), rgba(244,166,42,0) 65%);
        }

        .hero-band h2 {
            margin: 0;
            font-size: 25px;
            line-height: 1.2;
            font-weight: 920;
        }

        .hero-band p {
            margin: 8px 0 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.7;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 18px;
        }

        .insight-card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            min-height: 124px;
            box-shadow: 0 12px 28px rgba(23,33,26,.05);
        }

        .insight-card strong {
            display: block;
            font-size: 15px;
            margin-bottom: 8px;
        }

        .insight-card span {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.7;
        }

        .table-note {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border-radius: 8px;
            background: #EEF8F1;
            border: 1px solid #CBE3D0;
            color: #315642;
            margin: 8px 0 12px;
            font-size: 13px;
        }

        .subtle-card {
            background: #FBFDFB;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
        }

        .card-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }

        .card-title h3 {
            margin: 0;
            font-size: 17px;
            font-weight: 820;
        }

        .card-title span {
            color: var(--muted);
            font-size: 12px;
        }

        .mini-copy {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.7;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin: 4px 0 16px;
        }

        .metric-tile {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 15px 16px;
            box-shadow: 0 14px 32px rgba(31, 122, 77, .07);
            min-height: 108px;
        }

        .metric-tile .label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
        }

        .metric-tile .value {
            color: var(--ink);
            font-size: 30px;
            font-weight: 900;
            line-height: 1.2;
            margin-top: 8px;
        }

        .metric-tile .delta {
            color: var(--green);
            font-size: 12px;
            margin-top: 6px;
            font-weight: 700;
        }

        .compass {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .compass-item {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            background: #FFFFFF;
        }

        .compass-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            font-weight: 800;
        }

        .bar-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #E8EFE7;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
        }

        .action-list {
            display: grid;
            gap: 10px;
        }

        .action-item {
            display: grid;
            grid-template-columns: 9px 1fr;
            gap: 10px;
            padding: 12px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #FBFDFB;
        }

        .dot {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            margin-top: 6px;
        }

        .prob-card {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: #FFFFFF;
            padding: 20px;
            box-shadow: 0 18px 45px rgba(31,122,77,.08);
        }

        .prob-number {
            font-size: 54px;
            font-weight: 950;
            line-height: 1;
            color: var(--green);
        }

        .prob-label {
            display: inline-flex;
            margin-top: 12px;
            padding: 7px 10px;
            border-radius: 999px;
            color: white;
            font-weight: 800;
            font-size: 13px;
        }

        .login-title {
            font-size: 30px;
            line-height: 1.2;
            font-weight: 900;
            margin: 0 0 8px;
            color: var(--ink);
        }

        .login-copy {
            color: var(--muted);
            line-height: 1.7;
            margin-bottom: 22px;
        }

        .login-panel {
            background: rgba(255,255,255,.94);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 28px;
            box-shadow: 0 24px 65px rgba(23,33,26,.09);
        }

        .demo-account {
            padding: 11px 12px;
            border: 1px dashed #B9CDBD;
            border-radius: 8px;
            background: #F6FBF5;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.7;
            margin-top: 14px;
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 8px !important;
        }

        @media (max-width: 1000px) {
            .metric-strip,
            .compass,
            .insight-grid {
                grid-template-columns: 1fr;
            }
            .app-title {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


LOGIN_ANIMATION_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 640px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Noto Sans SC", sans-serif;
  background:
    linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px),
    linear-gradient(145deg, #10251A 0%, #1F7A4D 100%);
  background-size: 42px 42px, 42px 42px, auto;
  color: white;
  overflow: hidden;
}
.stage {
  min-height: 640px;
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 38px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 850;
  letter-spacing: 0;
  position: relative;
  z-index: 3;
}
.mark {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.22);
}
.copy {
  position: relative;
  z-index: 3;
  max-width: 520px;
}
.eyebrow {
  display: inline-flex;
  padding: 8px 11px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.22);
  color: rgba(255,255,255,.78);
  background: rgba(255,255,255,.08);
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 16px;
}
h1 {
  margin: 0;
  font-size: 40px;
  line-height: 1.08;
  letter-spacing: 0;
}
p {
  margin: 14px 0 0;
  color: rgba(255,255,255,.72);
  line-height: 1.8;
  font-size: 14px;
}
.characters {
  position: relative;
  z-index: 2;
  height: 255px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 18px;
  margin-top: 22px;
}
.blob {
  position: relative;
  width: 108px;
  border-radius: 32px 32px 24px 24px;
  box-shadow: inset 0 -18px 0 rgba(0,0,0,.12), 0 25px 45px rgba(0,0,0,.18);
  transform-origin: bottom center;
  animation: sway 4s ease-in-out infinite;
}
.blob.tall { height: 190px; background: #F4A62A; animation-delay: -.6s; }
.blob.mid { height: 150px; background: #E06C7F; animation-delay: -1.2s; }
.blob.short { height: 125px; background: #2B8C9C; animation-delay: -.2s; }
.blob.dark { height: 168px; background: #17211A; animation-delay: -1.8s; }
.face {
  position: absolute;
  top: 42px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 12px;
}
.eye {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: white;
  position: relative;
  overflow: hidden;
}
.pupil {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: #17211A;
  position: absolute;
  left: 9px;
  top: 10px;
  transition: transform .12s ease-out;
}
.mouth {
  position: absolute;
  top: 92px;
  left: 50%;
  width: 32px;
  height: 12px;
  border-bottom: 4px solid rgba(0,0,0,.45);
  border-radius: 0 0 24px 24px;
  transform: translateX(-50%);
}
.hands {
  position: absolute;
  top: 35px;
  left: 14px;
  right: 14px;
  height: 34px;
  display: flex;
  justify-content: space-between;
  opacity: 0;
  transform: translateY(18px);
  animation: peek 5s ease-in-out infinite;
}
.hand {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: rgba(255,255,255,.82);
}
.chips {
  position: absolute;
  inset: auto 38px 36px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  z-index: 4;
}
.chip {
  font-size: 12px;
  padding: 7px 9px;
  border-radius: 999px;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.18);
  color: rgba(255,255,255,.72);
}
@keyframes sway {
  0%, 100% { transform: translateY(0) skewX(0deg); }
  50% { transform: translateY(-8px) skewX(-2deg); }
}
@keyframes peek {
  0%, 42%, 100% { opacity: 0; transform: translateY(18px); }
  52%, 68% { opacity: 1; transform: translateY(0); }
}
</style>
</head>
<body>
  <div class="stage" id="stage">
    <div class="brand">
      <div class="mark">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M4 12C6.8 6.8 11.2 4.2 17.5 4.2C19 4.2 20.2 5.4 20.2 6.9C20.2 13.2 17.3 17.5 12 20C6.7 17.5 3.8 13.2 3.8 6.9C3.8 5.4 5 4.2 6.5 4.2C8.3 4.2 10 4.4 11.5 5" stroke="white" stroke-width="2" stroke-linecap="round" />
          <path d="M8 12.2L11 15.2L16.5 9.2" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <span>Repurchase OS</span>
    </div>

    <div class="copy">
      <span class="eyebrow">TabPFN · SHAP · 人货场运营</span>
      <h1>用模型找到<br/>下一次复购</h1>
      <p>把订单、商品、优惠、城市与用餐情境转成可解释的复购概率，帮助运营团队优先触达真正值得触达的人。</p>
      <div class="characters">
        <div class="blob mid">
          <div class="face"><div class="eye"><span class="pupil"></span></div><div class="eye"><span class="pupil"></span></div></div>
          <div class="mouth"></div>
        </div>
        <div class="blob tall">
          <div class="hands"><span class="hand"></span><span class="hand"></span></div>
          <div class="face"><div class="eye"><span class="pupil"></span></div><div class="eye"><span class="pupil"></span></div></div>
          <div class="mouth"></div>
        </div>
        <div class="blob dark">
          <div class="face"><div class="eye"><span class="pupil"></span></div><div class="eye"><span class="pupil"></span></div></div>
          <div class="mouth"></div>
        </div>
        <div class="blob short">
          <div class="face"><div class="eye"><span class="pupil"></span></div><div class="eye"><span class="pupil"></span></div></div>
          <div class="mouth"></div>
        </div>
      </div>
    </div>

    <div class="chips">
      <span class="chip">AUC 0.9217</span>
      <span class="chip">F1 0.8774</span>
      <span class="chip">Top10% Lift 1.5412</span>
      <span class="chip">货 51.40%</span>
    </div>
  </div>
<script>
const stage = document.getElementById("stage");
const pupils = Array.from(document.querySelectorAll(".pupil"));
stage.addEventListener("mousemove", (event) => {
  const rect = stage.getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  pupils.forEach((pupil) => {
    const box = pupil.parentElement.getBoundingClientRect();
    const ex = box.left - rect.left + box.width / 2;
    const ey = box.top - rect.top + box.height / 2;
    const angle = Math.atan2(cy - ey, cx - ex);
    const dist = Math.min(7, Math.hypot(cx - ex, cy - ey) / 26);
    pupil.style.transform = `translate(${Math.cos(angle) * dist}px, ${Math.sin(angle) * dist}px)`;
  });
});
</script>
</body>
</html>
"""


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="gbk")


def get_data(name: str) -> pd.DataFrame:
    return load_csv(str(DATA_FILES[name]))


def safe_image(path: Path, caption: Optional[str] = None, use_container_width: bool = True) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch" if use_container_width else "content")
    else:
        st.warning(f"未找到图片：{path.name}")


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def page_header(title: str, subtitle: str, meta: Optional[str] = None) -> None:
    meta_html = f'<span class="status-pill">{meta}</span>' if meta else ""
    st.markdown(
        f"""
        <div class="app-title">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="card-title">
            <h3>{title}</h3>
            <span>{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip(metrics: Iterable[Tuple[str, str, str]]) -> None:
    metric_list = list(metrics)
    columns = st.columns(len(metric_list))
    for column, (label, value, delta) in zip(columns, metric_list):
        column.markdown(
            f'<div class="metric-tile"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="delta">{delta}</div></div>',
            unsafe_allow_html=True,
        )


def render_compass() -> None:
    items = [
        ("货 Product", "51.40%", "#1F7A4D", 51.40, "品质、价格、包装、库存是复购的主轴"),
        ("人 Consumer", "41.09%", "#2B8C9C", 41.09, "年龄、支付方式、购买数量体现用户行为"),
        ("场 Context", "7.51%", "#F4A62A", 7.51, "城市级别、节假日、用餐高峰决定触达窗口"),
    ]
    columns = st.columns(3)
    for column, (title, value, color, width, desc) in zip(columns, items):
        column.markdown(
            f'<div class="compass-item"><div class="compass-top"><span>{title}</span>'
            f'<strong style="color:{color}">{value}</strong></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width}%; background:{color};"></div></div>'
            f'<p class="mini-copy" style="margin:12px 0 0;">{desc}</p></div>',
            unsafe_allow_html=True,
        )


def render_actions(actions: List[Tuple[str, str, str]]) -> None:
    for color, title, copy in actions:
        st.markdown(
            f'<div class="action-item"><span class="dot" style="background:{color};"></span>'
            f'<div><strong>{title}</strong><div class="mini-copy">{copy}</div></div></div>',
            unsafe_allow_html=True,
        )


def display_grid(
    df: pd.DataFrame,
    key: str,
    height: int = 420,
    selection: bool = False,
    fit_columns: bool = True,
    visible_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if visible_columns:
        df = df[[col for col in visible_columns if col in df.columns]].copy()
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
        wrapText=True,
        autoHeight=True,
    )
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_grid_options(
        rowHeight=46,
        headerHeight=44,
        suppressCellFocus=True,
        animateRows=True,
    )
    if selection:
        gb.configure_selection("multiple", use_checkbox=True, groupSelectsChildren=True)
    if fit_columns:
        gb.configure_grid_options(domLayout="normal")
    if "复购概率" in df.columns:
        gb.configure_column(
            "复购概率",
            type=["numericColumn"],
            valueFormatter=JsCode("function(params){return (params.value * 100).toFixed(1) + '%'}"),
            cellStyle=JsCode(
                """
                function(params) {
                    if (params.value >= 0.75) return {'color':'#1F7A4D','fontWeight':'900'};
                    if (params.value >= 0.45) return {'color':'#B7791F','fontWeight':'800'};
                    return {'color':'#E45745','fontWeight':'800'};
                }
                """
            ),
        )
    if "复购等级" in df.columns:
        gb.configure_column(
            "复购等级",
            cellStyle=JsCode(
                """
                function(params) {
                    if (params.value === '高潜力') return {'backgroundColor':'#E7F6EC','color':'#1F7A4D','fontWeight':'900'};
                    if (params.value === '可转化') return {'backgroundColor':'#FFF5DB','color':'#9A6415','fontWeight':'850'};
                    return {'backgroundColor':'#FDECE9','color':'#B7352A','fontWeight':'850'};
                }
                """
            ),
        )
    if "复购状态" in df.columns:
        gb.configure_column(
            "复购状态",
            cellStyle=JsCode(
                """
                function(params) {
                    if (params.value === '已复购') return {'color':'#1F7A4D','fontWeight':'850'};
                    return {'color':'#E45745','fontWeight':'850'};
                }
                """
            ),
        )
    if "模型" in df.columns:
        cell_style = JsCode(
            """
            function(params) {
                if (params.data && params.data['模型'] === 'TabPFN') {
                    return {'backgroundColor': '#EAF6EF', 'fontWeight': '800', 'color': '#1F7A4D'};
                }
                return {};
            }
            """
        )
        for col in df.columns:
            gb.configure_column(col, cellStyle=cell_style)
    return AgGrid(
        df,
        gridOptions=gb.build(),
        height=height,
        theme="fresh",
        update_on=["selectionChanged"] if selection else [],
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=fit_columns,
        show_search=True,
        show_download_button=True,
        custom_css={
            ".ag-root-wrapper": {"border-radius": "8px", "border": "1px solid #DDE7DD"},
            ".ag-header": {"background-color": "#EEF8F1", "border-bottom": "1px solid #CBE3D0"},
            ".ag-header-cell-text": {"font-weight": "800", "color": "#17211A"},
            ".ag-row": {"font-size": "13px"},
            ".ag-row-hover": {"background-color": "#F5FBF4 !important"},
            ".ag-cell": {"line-height": "1.45", "display": "flex", "align-items": "center"},
        },
        key=key,
    )


def format_orders_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "是否复购" in display.columns:
        display["复购状态"] = display["是否复购"].map({1: "已复购", 0: "未复购"})
    if "下单时间" in display.columns:
        display["下单时间"] = pd.to_datetime(display["下单时间"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    for col in ["单价(元)", "实际支付(元)", "优惠金额(元)"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
    return display


def render_table_note(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="table-note"><strong>{title}</strong><span>{copy}</span></div>',
        unsafe_allow_html=True,
    )


def render_insight_cards(cards: List[Tuple[str, str, str]]) -> None:
    html = "".join(
        f'<div class="insight-card"><strong style="color:{color};">{title}</strong><span>{copy}</span></div>'
        for title, copy, color in cards
    )
    st.markdown(f'<div class="insight-grid">{html}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def demo_cookie_key() -> str:
    return secrets.token_urlsafe(32)


def load_authenticator() -> Tuple[stauth.Authenticate, Dict[str, Any]]:
    config = {
        **AUTH_CONFIG,
        "cookie": {**AUTH_CONFIG["cookie"], "key": demo_cookie_key()},
    }
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        auto_hash=False,
    )
    return authenticator, config


def render_login(authenticator: stauth.Authenticate) -> None:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        login_src = "data:text/html;base64," + base64.b64encode(LOGIN_ANIMATION_HTML.encode("utf-8")).decode("ascii")
        st.iframe(login_src, height=660, width="stretch")
    with right:
        st.markdown(
            """
            <h1 class="login-title">登录到复购预测工作台</h1>
            <div class="login-copy">查看模型洞察、圈选高潜力用户，并把 SHAP 解释转成可执行的运营策略。</div>
            """,
            unsafe_allow_html=True,
        )
        login_result = authenticator.login(
            location="main",
            fields={
                "Form name": "账号认证",
                "Username": "账号",
                "Password": "密码",
                "Login": "进入工作台",
            },
            key="repurchase_login",
        )
        if login_result is not None:
            _, authentication_status, _ = login_result
        else:
            authentication_status = st.session_state.get("authentication_status")

        if authentication_status is False:
            st.error("账号或密码不正确，请重新输入。")
        elif authentication_status is None:
            st.info("请输入账号密码进入系统。")

        st.markdown(
            """
            <div class="demo-account">
            演示账号：admin / admin123
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner="正在准备预测引擎...")
def train_or_load_model() -> Tuple[Any, str]:
    artifact_path = ROOT / "artifacts" / "models" / "tabpfn.pkl"
    if artifact_path.exists():
        bundle = joblib.load(artifact_path)
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle, "TabPFN artifact"
        return {"model": bundle, "features": FEATURE_ORDER}, "TabPFN artifact"

    if TabPFNClassifier is None:
        return {"model": None, "features": FEATURE_ORDER, "error": "tabpfn 未安装"}, "TabPFN 指标模式"

    encoded = get_data("encoded")
    X = encoded[FEATURE_ORDER].copy()
    y = encoded["是否复购"].astype(int)

    try:
        model = TabPFNClassifier(
            n_estimators=4,
            device="cpu",
            random_state=42,
            show_progress_bar=False,
            ignore_pretraining_limits=True,
        )
        model.fit(X, y)
        return {"model": model, "features": FEATURE_ORDER}, "TabPFNClassifier"
    except Exception as exc:
        return {"model": None, "features": FEATURE_ORDER, "error": str(exc)}, "TabPFN 指标模式"

def tabpfn_proxy_probability(encoded_df: pd.DataFrame) -> np.ndarray:
    x = encoded_df.copy()
    rating = (x["商品评分"].astype(float) - 37.0) / 8.6
    stock_risk = np.clip((500.0 - x["库存剩余(件)"].astype(float)) / 1200.0, -1.2, 1.2)
    unit_price = (x["单价(元)"].astype(float) - 29.7) / 11.5
    discount = np.clip(x["优惠金额(元)"].astype(float) / 18.0, 0, 2.0)
    age_core = x["客户年龄段"].isin([1, 2]).astype(float)
    peak = x["是否用餐高峰"].astype(float)
    promo = x["优惠方式"].isin([3, 4, 5]).astype(float)
    meat_high_price = ((x["商品类别"].eq(3)) & (x["单价(元)"].astype(float) > 40)).astype(float)
    score = (
        0.55
        + 0.52 * rating
        - 0.30 * stock_risk
        - 0.22 * unit_price
        + 0.18 * discount
        + 0.20 * age_core
        + 0.12 * peak
        + 0.10 * promo
        - 0.28 * meat_high_price
    )
    return 1 / (1 + np.exp(-score))


def encode_record(record: Dict[str, Any]) -> pd.DataFrame:
    encoded = {}
    for feature in FEATURE_ORDER:
        value = record.get(feature)
        if feature in ENCODERS:
            encoded[feature] = ENCODERS[feature].get(str(value), 0)
        else:
            encoded[feature] = float(value)
    return pd.DataFrame([encoded], columns=FEATURE_ORDER)


def predict_probability(encoded_df: pd.DataFrame) -> np.ndarray:
    bundle, _ = train_or_load_model()
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    features = bundle.get("features", FEATURE_ORDER) if isinstance(bundle, dict) else FEATURE_ORDER
    X = encoded_df[features]
    if model is not None and hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if model is None:
        return tabpfn_proxy_probability(X)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def probability_level(prob: float) -> Tuple[str, str, str]:
    if prob >= 0.75:
        return "高潜力", "#1F7A4D", "优先小额激励与会员权益，不必过度补贴。"
    if prob >= 0.45:
        return "可转化", "#F4A62A", "适合叠加满减券、套餐组合和高峰时段提醒。"
    return "流失风险", "#E45745", "建议先排查价格、库存、评分体验，再做较强优惠。"


def rule_drivers(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    positive = []
    negative = []
    rating = float(record["商品评分"])
    discount = float(record["优惠金额(元)"])
    stock = float(record["库存剩余(件)"])
    price = float(record["单价(元)"])
    actual = float(record["实际支付(元)"])
    category = str(record["商品类别"])
    city = str(record["消费城市级别"])
    age = str(record["客户年龄段"])

    if rating >= 42:
        positive.append("商品评分高，品质信任会明显推高复购概率。")
    elif rating < 32:
        negative.append("商品评分偏低，单纯优惠很难抵消体验风险。")

    if discount >= 10 and rating >= 38:
        positive.append("高评分叠加 10 元以上优惠，属于强转化组合。")
    elif discount == 0:
        negative.append("当前无优惠刺激，中等概率用户可能缺少复购触发点。")

    if stock < 500:
        negative.append("库存低于 500 件，可能放大缺货预期与流失风险。")
    elif stock >= 4000:
        positive.append("库存充足，适合承接批量触达和高峰订单。")

    if city == "三线" and 100 <= actual <= 200:
        positive.append("三线城市 100-200 元实际支付区间更贴近性价比复购窗口。")

    if category == "肉食" and price > 40:
        negative.append("肉食类单价高于 40 元阈值，建议用套餐或满减降低心理价格。")

    if age in ["25-34", "35-44"]:
        positive.append("核心年龄段对便捷化、品质稳定的预制菜复购更敏感。")

    if record["是否用餐高峰"] == "是":
        positive.append("用餐高峰触达窗口明确，适合即时推送。")

    return positive[:4], negative[:4]


def infer_category(product_name: str) -> str:
    name = str(product_name)
    if any(token in name for token in ["豆腐", "豆干", "腐竹"]):
        return "豆制品"
    if any(token in name for token in ["面", "饭", "汤", "粥", "粉"]):
        return "主食面汤"
    if any(token in name for token in ["蛋", "菜", "笋", "菇", "藕"]):
        return "蔬菜"
    return "肉食"


def parse_package_weight(value: Any) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else 500.0


def infer_peak_hour(timestamp: Any) -> str:
    try:
        hour = pd.to_datetime(timestamp).hour
    except Exception:
        return "否"
    return "是" if hour in list(range(11, 14)) + list(range(17, 20)) else "否"


def infer_holiday(timestamp: Any) -> str:
    try:
        weekday = pd.to_datetime(timestamp).weekday()
    except Exception:
        return "否"
    return "是" if weekday >= 5 else "否"


def prepare_raw_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame()
    prepared["单价(元)"] = df.get("单价(元)", 29.72)
    prepared["数量(个)"] = df.get("数量(个)", 5)
    prepared["消费城市级别"] = df.get("消费城市级别", "新一线")
    prepared["支付方式"] = df.get("支付方式", "支付宝")
    prepared["优惠方式"] = df.get("优惠方式", "会员折扣")
    prepared["优惠金额(元)"] = df.get("优惠金额(元)", 0)
    prepared["实际支付(元)"] = df.get("实际支付(元)", prepared["单价(元)"] * prepared["数量(个)"] - prepared["优惠金额(元)"])
    prepared["库存剩余(件)"] = df.get("库存剩余(件)", 4000)
    prepared["商品评分"] = df.get("商品评分", 38)
    prepared["客户年龄段"] = df.get("客户年龄段", "25-34")
    prepared["商品类别"] = df["商品名称"].map(infer_category) if "商品名称" in df.columns else "肉食"
    prepared["包装规格(g)"] = df["包装规格"].map(parse_package_weight) if "包装规格" in df.columns else 500
    prepared["是否节假日"] = df["下单时间"].map(infer_holiday) if "下单时间" in df.columns else "否"
    prepared["是否用餐高峰"] = df["下单时间"].map(infer_peak_hour) if "下单时间" in df.columns else "否"
    return prepared


def encode_batch(prepared: pd.DataFrame) -> pd.DataFrame:
    encoded = prepared.copy()
    for col, mapping in ENCODERS.items():
        encoded[col] = encoded[col].map(mapping).fillna(0).astype(int)
    for col in FEATURE_ORDER:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(0)
    return encoded[FEATURE_ORDER]


def add_prediction_columns(raw: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_raw_for_prediction(raw)
    probs = predict_probability(encode_batch(prepared))
    result = raw.copy()
    result["复购概率"] = np.round(probs, 4)
    result["复购等级"] = [probability_level(float(p))[0] for p in probs]
    result["建议动作"] = [
        "会员权益 + 小额券" if p >= 0.75 else "满减券 + 高峰提醒" if p >= 0.45 else "强优惠 + 体验修复"
        for p in probs
    ]
    return result.sort_values("复购概率", ascending=False)


def render_dashboard() -> None:
    raw = get_data("raw")
    metrics = get_data("metrics")
    tabpfn = metrics.loc[metrics["模型"].eq("TabPFN")].iloc[0]
    repurchase_rate = raw["是否复购"].mean()
    model_bundle, model_mode = train_or_load_model()
    model_status = (
        f"TabPFN · {model_mode}"
        if model_bundle.get("model") is not None
        else "TabPFN 实验结果 · 单笔规则演示"
    )

    page_header(
        "总览",
        "一屏查看复购大盘、TabPFN 实验结果、关键解释和运营动作。",
        model_status,
    )

    st.markdown(
        """
        <div class="hero-band">
            <h2>预制菜复购预测运营工作台</h2>
            <p>基于江浙沪 2217 条订单，围绕商品评分、库存、价格、优惠和消费场景，生成可解释的复购判断。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_strip(
        [
            ("样本订单", f"{len(raw):,}", "江浙沪订单数据"),
            ("复购率", format_pct(repurchase_rate), "目标变量基线"),
            ("TabPFN AUC", f"{tabpfn['AUC']:.4f}", "区分能力最优"),
            ("TabPFN F1", f"{tabpfn['F1分数']:.4f}", "精确率与召回平衡"),
            ("Top10% Lift", f"{tabpfn['Top10%提升度']:.4f}", "营销效率提升"),
        ]
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.metric_card(title="准确率", content=f"{tabpfn['准确率']:.2%}", description="TabPFN 测试集表现", key="m_acc")
    with k2:
        ui.metric_card(title="召回率", content=f"{tabpfn['召回率']:.2%}", description="尽量不漏掉复购用户", key="m_recall")
    with k3:
        ui.metric_card(title="MCC", content=f"{tabpfn['MCC系数']:.4f}", description="不平衡数据稳健指标", key="m_mcc")
    with k4:
        ui.metric_card(title="Brier", content=f"{tabpfn['Brier分数']:.4f}", description="概率校准误差更低更好", key="m_brier")

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section_title("人-货-场三维罗盘", "SHAP 聚合贡献")
        render_compass()
    with right:
        section_title("今日优先动作", "运营可执行")
        render_actions(
            [
                ("#1F7A4D", "先做高评分商品复购放大", "商品评分是首要解释特征，优先把优惠给体验已被验证的商品。"),
                ("#F4A62A", "盯住低库存高概率用户", "库存低于 500 件时，复购机会可能被缺货预期吞掉。"),
                ("#2B8C9C", "三线城市做性价比套餐", "100-200 元实际支付区间更适合三线城市复购转化。"),
            ]
        )

    render_insight_cards(
        [
            ("产品评分是第一抓手", "评分高的商品更适合小额权益放大，而不是一味大额补贴。", "#1F7A4D"),
            ("库存不足会吞掉机会", "高概率用户遇到低库存，运营应该优先补货或推替代商品。", "#E45745"),
            ("优惠要绑定品质", "高评分 + 10 元以上优惠是强组合，低评分商品降价效果有限。", "#F4A62A"),
        ]
    )

    chart_col, table_col = st.columns([1, 1], gap="large")
    with chart_col:
        section_title("目标变量分布", "复购/未复购")
        safe_image(IMAGE_FILES["target"])
    with table_col:
        section_title("TabPFN 核心指标", "只展示主模型")
        tab_table = pd.DataFrame(
            [
                {"指标": "准确率", "数值": f"{tabpfn['准确率']:.2%}", "业务含义": "整体判断正确率"},
                {"指标": "召回率", "数值": f"{tabpfn['召回率']:.2%}", "业务含义": "尽量不漏掉潜在复购用户"},
                {"指标": "AUC", "数值": f"{tabpfn['AUC']:.4f}", "业务含义": "区分复购/未复购能力"},
                {"指标": "Top10% Lift", "数值": f"{tabpfn['Top10%提升度']:.4f}", "业务含义": "前 10% 用户筛选效率"},
            ]
        )
        display_grid(tab_table, key="dashboard_tabpfn_metrics", height=260, fit_columns=True)


def render_data_assets() -> None:
    page_header("数据与模型", "用更清晰的业务表格查看订单数据，并聚焦 TabPFN 主模型表现。", "表格已精简字段")
    selected = ui.tabs(
        options=["订单明细", "TabPFN 模型", "字段口径"],
        default_value="订单明细",
        key="data_asset_tabs",
    )
    selected = selected or "订单明细"

    if selected == "订单明细":
        df = format_orders_table(get_data("raw"))
        c1, c2, c3, c4 = st.columns(4)
        city = c1.multiselect("城市级别", sorted(df["消费城市级别"].dropna().unique()))
        age = c2.multiselect("年龄段", sorted(df["客户年龄段"].dropna().unique()))
        promo = c3.multiselect("优惠方式", sorted(df["优惠方式"].dropna().unique()))
        target = c4.selectbox("复购标签", ["全部", "已复购", "未复购"])
        filtered = df.copy()
        if city:
            filtered = filtered[filtered["消费城市级别"].isin(city)]
        if age:
            filtered = filtered[filtered["客户年龄段"].isin(age)]
        if promo:
            filtered = filtered[filtered["优惠方式"].isin(promo)]
        if target != "全部":
            filtered = filtered[filtered["复购状态"].eq(target)]
        ui.badges(
            badge_list=[
                (f"{len(filtered):,} 条记录", "default"),
                (f"复购率 {filtered['复购状态'].eq('已复购').mean():.2%}" if len(filtered) else "复购率 -", "secondary"),
                ("支持搜索/筛选/导出", "outline"),
            ],
            key="raw_badges",
        )
        render_table_note("清晰字段", "只展示运营最常看的订单、用户、商品、金额、优惠、库存、评分和复购状态。")
        display_grid(
            filtered,
            key="raw_grid",
            height=520,
            visible_columns=[
                "订单ID",
                "用户ID",
                "商品名称",
                "下单时间",
                "消费城市级别",
                "客户年龄段",
                "优惠方式",
                "实际支付(元)",
                "库存剩余(件)",
                "商品评分",
                "复购状态",
            ],
        )

    elif selected == "TabPFN 模型":
        metrics = get_data("metrics")
        tabpfn = metrics.loc[metrics["模型"].eq("TabPFN")].iloc[0]
        render_metric_strip(
            [
                ("准确率", f"{tabpfn['准确率']:.2%}", "整体表现"),
                ("精确率", f"{tabpfn['精确率']:.2%}", "减少误投"),
                ("召回率", f"{tabpfn['召回率']:.2%}", "减少漏判"),
                ("AUC", f"{tabpfn['AUC']:.4f}", "区分能力"),
                ("Top10% Lift", f"{tabpfn['Top10%提升度']:.4f}", "营销效率"),
            ]
        )
        tab_table = pd.DataFrame(
            [
                {"指标": "F1 分数", "TabPFN": f"{tabpfn['F1分数']:.4f}", "说明": "综合精确率和召回率"},
                {"指标": "MCC 系数", "TabPFN": f"{tabpfn['MCC系数']:.4f}", "说明": "适合不平衡数据的稳健指标"},
                {"指标": "Brier 分数", "TabPFN": f"{tabpfn['Brier分数']:.4f}", "说明": "概率校准误差，越低越好"},
                {"指标": "Top20% Lift", "TabPFN": f"{tabpfn['Top20%提升度']:.4f}", "说明": "前 20% 用户筛选效率"},
            ]
        )
        display_grid(tab_table, key="tabpfn_only_table", height=230, fit_columns=True)
        section_title("扇形混淆矩阵", "仅展示扇形图")
        safe_image(IMAGE_FILES["matrix"], "TabPFN 与模型对比的扇形混淆矩阵图")

    else:
        mapping = get_data("mapping")
        encoded = get_data("encoded")
        raw = get_data("raw")
        ui.badges(
            badge_list=[(f"原始订单 {raw.shape[0]} 行", "default"), (f"建模字段 {encoded.shape[1]} 列", "secondary"), ("自动编码", "outline")],
            key="schema_badges",
        )
        render_insight_cards(
            [
                ("人", "客户年龄段、支付方式、购买数量、评分行为。", "#2B8C9C"),
                ("货", "商品类别、单价、包装规格、实际支付、优惠、库存。", "#1F7A4D"),
                ("场", "城市级别、节假日、用餐高峰。", "#F4A62A"),
            ]
        )
        section_title("编码映射", "中文业务字段到模型编码")
        display_grid(mapping, key="mapping_grid", height=260, fit_columns=True)


def render_single_predict() -> None:
    bundle, model_mode = train_or_load_model()
    mode_label = f"TabPFN 实时预测 · {model_mode}" if bundle.get("model") is not None else "单笔规则演示"
    page_header("复购预测", "输入一个订单画像，输出复购概率、等级、驱动因素和运营建议。", mode_label)

    if bundle.get("model") is None:
        st.info("此在线演示未加载 TabPFN 模型权重；单笔概率由规则化估计生成。页面中的 TabPFN 指标和 SHAP 图来自既有实验结果，不代表该规则估计器的验证性能。")

    left, right = st.columns([1, 0.95], gap="large")
    with left:
        with st.form("single_predict_form"):
            st.subheader("订单与用户画像")
            c1, c2, c3 = st.columns(3)
            product_category = c1.selectbox("商品类别", list(ENCODERS["商品类别"].keys()), index=3)
            city_tier = c2.selectbox("消费城市级别", list(ENCODERS["消费城市级别"].keys()), index=1)
            age_group = c3.selectbox("客户年龄段", list(ENCODERS["客户年龄段"].keys()), index=1)

            c4, c5, c6 = st.columns(3)
            payment = c4.selectbox("支付方式", list(ENCODERS["支付方式"].keys()), index=3)
            promo = c5.selectbox("优惠方式", list(ENCODERS["优惠方式"].keys()), index=5)
            package_weight = c6.select_slider("包装规格(g)", options=[200, 300, 400, 500, 600, 800, 1000], value=500)

            c7, c8, c9 = st.columns(3)
            unit_price = c7.slider("单价(元)", 10.0, 70.0, 29.7, 0.1)
            quantity = c8.slider("数量(个)", 1, 10, 5)
            discount = c9.slider("优惠金额(元)", 0.0, 90.0, 10.0, 1.0)

            c10, c11, c12 = st.columns(3)
            actual_payment = c10.number_input("实际支付(元)", min_value=0.0, max_value=600.0, value=max(unit_price * quantity - discount, 0.0), step=1.0)
            stock = c11.slider("库存剩余(件)", 0, 10000, 4200, 50)
            rating = c12.slider("商品评分", 10, 50, 42, 1)

            c13, c14 = st.columns(2)
            holiday = c13.selectbox("是否节假日", ["否", "是"], index=0)
            peak = c14.selectbox("是否用餐高峰", ["否", "是"], index=1)

            submitted = st.form_submit_button("生成复购预测", width="stretch")

        record = {
            "单价(元)": unit_price,
            "数量(个)": quantity,
            "消费城市级别": city_tier,
            "支付方式": payment,
            "优惠方式": promo,
            "优惠金额(元)": discount,
            "实际支付(元)": actual_payment,
            "库存剩余(件)": stock,
            "商品评分": rating,
            "客户年龄段": age_group,
            "商品类别": product_category,
            "包装规格(g)": package_weight,
            "是否节假日": holiday,
            "是否用餐高峰": peak,
        }

    with right:
        if submitted or "last_single_record" in st.session_state:
            if submitted:
                st.session_state["last_single_record"] = record
            record = st.session_state["last_single_record"]
            prob = float(predict_probability(encode_record(record))[0])
            level, color, advice = probability_level(prob)
            positives, negatives = rule_drivers(record)

            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="mini-copy">预测复购概率</div>
                    <div class="prob-number">{prob:.1%}</div>
                    <span class="prob-label" style="background:{color};">{level}</span>
                    <p class="mini-copy" style="margin-top:14px;">{advice}</p>
                    <div class="bar-track" style="margin-top:18px;"><div class="bar-fill" style="width:{prob*100:.1f}%; background:{color};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            section_title("解释摘要", "规则化业务翻译")
            if positives:
                st.success("正向因素：" + "；".join(positives))
            if negatives:
                st.error("风险因素：" + "；".join(negatives))
            if not positives and not negatives:
                st.info("该样本处于中性区间，建议结合优惠和用餐高峰做一次轻触达。")

            values = [
                ("商品评分", (float(record["商品评分"]) - 30) / 20),
                ("优惠金额", float(record["优惠金额(元)"]) / 90),
                ("库存保障", min(float(record["库存剩余(件)"]) / 8000, 1)),
                ("价格压力", -max((float(record["单价(元)"]) - 40) / 30, 0)),
                ("高峰触达", 0.25 if record["是否用餐高峰"] == "是" else -0.15),
            ]
            fig = go.Figure(
                go.Bar(
                    x=[v for _, v in values],
                    y=[k for k, _ in values],
                    orientation="h",
                    marker_color=["#1F7A4D" if v >= 0 else "#E45745" for _, v in values],
                )
            )
            fig.update_layout(
                height=285,
                margin=dict(l=10, r=10, t=18, b=10),
                xaxis_title="相对影响方向",
                yaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            section_title("等待输入", "预测结果将在这里出现")
            st.markdown('<p class="mini-copy">填写左侧画像后点击“生成复购预测”，系统会输出概率、等级、正负驱动因素与建议动作。</p>', unsafe_allow_html=True)


def render_shap_center() -> None:
    page_header("解释策略", "把 TabPFN 的 SHAP 解释转成商品、库存、优惠和触达动作。", "解释 + 策略合一")

    selected = ui.tabs(
        options=["全局贡献", "蜂巢图", "交互依赖", "策略建议", "样本解释"],
        default_value="全局贡献",
        key="shap_tabs",
    )
    selected = selected or "全局贡献"

    if selected == "全局贡献":
        col1, col2 = st.columns([1.25, 0.75], gap="large")
        with col1:
            safe_image(IMAGE_FILES["dimension"], "TabPFN SHAP 维度贡献全景图")
        with col2:
            section_title("业务解读", "人-货-场")
            render_actions(
                [
                    ("#1F7A4D", "货：51.40%", "产品属性与价值是核心，商品评分、库存和单价是首要抓手。"),
                    ("#2B8C9C", "人：41.09%", "年龄段、支付方式、购买数量决定触达方式与权益表达。"),
                    ("#F4A62A", "场：7.51%", "城市级别与用餐高峰更像触达窗口，而不是单独决定因素。"),
                ]
            )

    elif selected == "蜂巢图":
        safe_image(IMAGE_FILES["beeswarm"], "TabPFN SHAP 蜂巢图")
        st.info("横轴越靠右代表越促进复购；颜色代表特征取值高低，可用于识别高评分、优惠、价格等变量的方向性。")

    elif selected == "交互依赖":
        safe_image(IMAGE_FILES["dependence"], "SHAP 交互依赖图")
        render_actions(
            [
                ("#1F7A4D", "产品评分 × 优惠金额", "高评分商品在 10 元以上优惠下复购促进效应更强。"),
                ("#2B8C9C", "实际支付 × 城市级别", "三线城市 100-200 元支付区间更容易形成复购窗口。"),
                ("#F4A62A", "年龄段 × 商品评分", "35-44 岁用户对评分更敏感，适合品质背书型文案。"),
            ]
        )

    elif selected == "策略建议":
        strategies = [
            ("高评分商品复购放大", "商品评分高且复购概率 >= 0.75", "会员权益 + 小额券 + 新品优先体验", "#1F7A4D"),
            ("库存风险拦截", "复购概率高但库存 < 500 件", "补货提醒 + 替代商品推荐 + 到货通知", "#E45745"),
            ("三线城市性价比套餐", "三线城市且实际支付 100-200 元", "套餐组合 + 满减券 + 包邮门槛", "#F4A62A"),
            ("核心年龄段品质背书", "25-34 / 35-44 岁且评分敏感", "口味稳定、快速出餐、真实评价文案", "#2B8C9C"),
        ]
        cols = st.columns(2)
        for idx, (title, condition, action, color) in enumerate(strategies):
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <div class="card-title"><h3 style="color:{color};">{title}</h3><span>策略 {idx + 1}</span></div>
                        <div class="mini-copy"><strong>触发条件：</strong>{condition}</div>
                        <div class="mini-copy" style="margin-top:8px;"><strong>推荐动作：</strong>{action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        force_files = sorted((ROOT / "图片").glob("SHAP_单样本力图_*.png"))
        if not force_files:
            st.warning("未找到单样本力图。")
            return
        labels = [p.stem.replace("SHAP_单样本力图_", "样本 ") for p in force_files]
        choice = st.selectbox("选择样本力图", labels, index=min(7, len(labels) - 1))
        path = force_files[labels.index(choice)]
        safe_image(path, choice)
        st.caption("红色表示推高复购概率的特征，蓝色表示压低复购概率的特征。")


def render_shell(authenticator: stauth.Authenticate, config: Dict[str, Any]) -> None:
    name = st.session_state.get("name", "用户")
    allowed_pages = APP_PAGES

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 4px 18px;">
              <div style="font-size: 18px; font-weight: 900; color: white;">复购预测工作台</div>
              <div style="font-size: 12px; color: rgba(255,255,255,.58); margin-top: 6px;">Prepared Food Repurchase OS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = option_menu(
            None,
            allowed_pages,
            icons=[PAGE_ICONS[p] for p in allowed_pages],
            default_index=0,
            styles={
                "container": {"padding": "6px", "background-color": "#0F2118", "border-radius": "8px"},
                "icon": {"color": "#DDEADF", "font-size": "17px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "4px 0",
                    "border-radius": "8px",
                    "color": "#DDEADF",
                    "background-color": "#0F2118",
                    "--hover-color": "#173C2A",
                },
                "nav-link-selected": {"background-color": "#1F7A4D", "color": "white"},
            },
        )

        st.markdown(
            f"""
            <div style="margin-top: 18px; padding: 12px; border: 1px solid rgba(255,255,255,.14); border-radius: 8px; background: rgba(255,255,255,.06);">
              <div style="font-weight: 800;">{name}</div>
              <div style="font-size: 12px; color: rgba(255,255,255,.62);">单账号演示模式</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        authenticator.logout("退出登录", location="sidebar", key="logout_button", use_container_width=True)

    if selected == "总览":
        render_dashboard()
    elif selected == "复购预测":
        render_single_predict()
    elif selected == "数据与模型":
        render_data_assets()
    elif selected == "解释策略":
        render_shap_center()


def main() -> None:
    inject_theme()
    authenticator, config = load_authenticator()
    if st.session_state.get("authentication_status") is True:
        render_shell(authenticator, config)
    else:
        render_login(authenticator)


if __name__ == "__main__":
    main()
