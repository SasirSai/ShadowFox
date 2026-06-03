import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS — clean light theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F7F6F3;
    --bg-card: #FFFFFF;
    --bg-sidebar: #FDFCFA;
    --terracotta: #C4553A;
    --terracotta-light: #F4E8E4;
    --sage: #3D7A5F;
    --sage-light: #E3F0EA;
    --plum: #7B4D8E;
    --plum-light: #F0E6F4;
    --sand: #B8860B;
    --sand-light: #FBF5E6;
    --text-dark: #1C1C1E;
    --text-mid: #4A4A52;
    --text-light: #8E8E93;
    --border: #E8E6E1;
    --border-hover: #D1CFC9;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 30px rgba(0,0,0,0.08);
    --radius: 10px;
    --radius-lg: 14px;
}

/* ═══════ Global ═══════ */
.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

html, body, .stApp, .stApp * {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ═══════ Sidebar ═══════ */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] label {
    color: var(--text-mid) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] input {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 3px var(--terracotta-light) !important;
}

/* ═══════ Headings ═══════ */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--text-mid) !important;
}

/* ═══════ Buttons ═══════ */
.stButton > button {
    background: var(--terracotta) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: #B04A32 !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ═══════ Tabs ═══════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-light) !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--terracotta) !important;
    border-bottom: 2px solid var(--terracotta) !important;
}

/* ═══════ Metrics ═══════ */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 22px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-light) !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    font-size: 0.7rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-dark) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* ═══════ Selectbox ═══════ */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ═══════ Dataframe ═══════ */
.stDataFrame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ═══════ Custom classes ═══════ */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
}

.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 32px 36px;
    text-align: center;
    box-shadow: var(--shadow-md);
    border-top: 3px solid var(--terracotta);
}
.result-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-light);
    margin-bottom: 8px;
}
.result-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--terracotta);
    margin: 0;
    line-height: 1.2;
}
.result-subtitle {
    font-size: 0.8rem;
    color: var(--text-light);
    margin-top: 10px;
}

/* ═══════ Gauge ═══════ */
.gauge-track {
    background: #EEECEA;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 20px;
}
.gauge-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--sage), var(--terracotta));
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-light);
}

/* ═══════ Stat row ═══════ */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.stat-item {
    flex: 1;
    min-width: 110px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-dark);
}
.stat-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-light);
    margin-top: 4px;
}

/* ═══════ Feature bar ═══════ */
.feat-bar-container { margin: 8px 0; }
.feat-bar-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.feat-bar-name {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-dark);
}
.feat-bar-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-light);
}
.feat-bar-track {
    background: #EEECEA;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* ═══════ Section label ═══════ */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ═══════ Divider ═══════ */
.divider {
    height: 1px;
    background: var(--border);
    margin: 24px 0;
}

/* ═══════ Sidebar group header ═══════ */
.sidebar-group {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
    padding: 6px 0 6px 10px;
    border-left: 3px solid;
}

/* ═══════ Tag ═══════ */
.tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 16px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.tag-terracotta { background: var(--terracotta-light); color: var(--terracotta); }
.tag-sage       { background: var(--sage-light);       color: var(--sage); }
.tag-plum       { background: var(--plum-light);       color: var(--plum); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Feature metadata
# ──────────────────────────────────────────────
FEATURE_INFO = {
    "CRIM":    {"label": "Crime Rate",          "desc": "Per capita crime rate",                            "group": "location"},
    "ZN":      {"label": "Residential Zone",    "desc": "Proportion of land zoned for large lots",          "group": "location"},
    "INDUS":   {"label": "Industrial Area",     "desc": "Proportion of non-retail business acres",          "group": "location"},
    "CHAS":    {"label": "Charles River",       "desc": "Borders the river (1 = Yes, 0 = No)",              "group": "location"},
    "NOX":     {"label": "Nitric Oxide",        "desc": "Concentration (parts per 10M)",                    "group": "environment"},
    "RM":      {"label": "Avg. Rooms",          "desc": "Average number of rooms per dwelling",             "group": "property"},
    "AGE":     {"label": "Building Age",        "desc": "% of units built before 1940",                     "group": "property"},
    "DIS":     {"label": "Distance to Jobs",    "desc": "Weighted distance to employment centres",          "group": "location"},
    "RAD":     {"label": "Highway Access",      "desc": "Accessibility index to radial highways",           "group": "location"},
    "TAX":     {"label": "Property Tax",        "desc": "Tax rate per $10,000",                             "group": "financial"},
    "PTRATIO": {"label": "Pupil-Teacher Ratio", "desc": "Ratio in the area",                               "group": "social"},
    "B":       {"label": "Demographics Index",  "desc": "1000(Bk - 0.63)^2 statistic",                     "group": "social"},
    "LSTAT":   {"label": "Lower Status %",      "desc": "% lower-status population",                       "group": "social"},
}

GROUP_META = {
    "property":    {"title": "Property",        "color": "var(--terracotta)"},
    "location":    {"title": "Location",        "color": "var(--sage)"},
    "environment": {"title": "Environment",     "color": "var(--plum)"},
    "financial":   {"title": "Financial",       "color": "var(--sand)"},
    "social":      {"title": "Community",       "color": "var(--text-mid)"},
}

BAR_COLORS = [
    "#C4553A", "#3D7A5F", "#7B4D8E", "#B8860B", "#5A7D9A",
    "#D4603A", "#2D6A4F", "#8E5FA0", "#C9971A", "#6B8DAA",
    "#A94430", "#1F5C43", "#6A3D7C",
]

# ──────────────────────────────────────────────
# Load and train
# ──────────────────────────────────────────────
@st.cache_resource
def load_and_train_model():
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    model_pipeline.fit(X, y)

    importances = model_pipeline.named_steps['regressor'].feature_importances_
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=False)

    return model_pipeline, list(X.columns), df, imp_df


with st.spinner("Loading model..."):
    model, feature_order, df, importance_df = load_and_train_model()

medians = df.median()
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 6px;">
    <span class="tag tag-terracotta">v2.0</span>
    <span class="tag tag-sage" style="margin-left: 4px;">Random Forest</span>
    <span class="tag tag-plum" style="margin-left: 4px;">506 samples</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="margin: 0 0 4px 0; font-size: 2rem; font-weight: 800; color: #1C1C1E;">
    House Price Predictor
</h1>
<p style="font-size: 0.95rem; color: #8E8E93; max-width: 620px; line-height: 1.6; margin-bottom: 0;">
    An ML-powered tool trained on the Boston Housing dataset. Configure the features in the sidebar and generate a prediction.
</p>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h3 style="font-size: 1rem; font-weight: 700; color: #1C1C1E; margin-bottom: 4px;">Feature Controls</h3>
    <p style="font-size: 0.78rem; color: #8E8E93; margin-bottom: 12px;">Adjust parameters for prediction</p>
    """, unsafe_allow_html=True)

    user_input = {}

    # Build grouped feature lists, but we'll track values for ALL features
    groups = {}
    for feat in feature_order:
        info = FEATURE_INFO.get(feat, {"label": feat, "desc": "", "group": "other"})
        g = info["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(feat)

    for group_key, feats in groups.items():
        meta = GROUP_META.get(group_key, {"title": group_key.title(), "color": "var(--text-mid)"})
        st.markdown(
            f'<div class="sidebar-group" style="border-color: {meta["color"]}; color: {meta["color"]};">{meta["title"]}</div>',
            unsafe_allow_html=True,
        )

        for feat in feats:
            info = FEATURE_INFO.get(feat, {"label": feat, "desc": ""})
            default_val = float(medians[feat]) if not pd.isna(medians[feat]) else 0.0
            min_val = float(X[feat].min()) if not pd.isna(X[feat].min()) else 0.0
            max_val = float(X[feat].max()) if not pd.isna(X[feat].max()) else 100.0

            if feat == "CHAS":
                user_input[feat] = float(st.selectbox(
                    info["label"],
                    options=[0, 1],
                    index=0,
                    help=info["desc"],
                ))
            elif feat == "RAD":
                user_input[feat] = float(st.slider(
                    info["label"],
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    help=info["desc"],
                ))
            else:
                user_input[feat] = st.number_input(
                    info["label"],
                    value=default_val,
                    help=info["desc"],
                    format="%.4f" if max_val < 2 else "%.2f",
                )

# ──────────────────────────────────────────────
# Main — tabs
# ──────────────────────────────────────────────
tab_predict, tab_insights, tab_data = st.tabs(["Predict", "Insights", "Data Explorer"])

# ──────────── TAB 1: PREDICT ────────────
with tab_predict:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("""
        <div class="card">
            <p class="section-label" style="color: var(--terracotta);">Prediction Engine</p>
            <p style="font-size: 0.88rem; line-height: 1.7; color: #4A4A52;">
                Configure the property features using the sidebar controls, then generate a prediction.
                The model uses a 200-tree Random Forest ensemble trained on the complete dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        predict_clicked = st.button("Generate Prediction", type="primary", use_container_width=True)

        if predict_clicked:
            # Build input DataFrame in the EXACT order the model was trained on
            ordered_input = {feat: user_input[feat] for feat in feature_order}
            input_df = pd.DataFrame([ordered_input], columns=feature_order)
            prediction = model.predict(input_df)[0]
            price = prediction * 1000

            price_min = y.min() * 1000
            price_max = y.max() * 1000
            pct = min(max((price - price_min) / (price_max - price_min) * 100, 0), 100)

            st.markdown(f"""
            <div class="result-card">
                <p class="result-label">Estimated Market Value</p>
                <p class="result-price">${price:,.0f}</p>
                <p class="result-subtitle">Based on {len(df)} training samples</p>
                <div class="gauge-track">
                    <div class="gauge-fill" style="width: {pct:.1f}%;"></div>
                </div>
                <div class="gauge-labels">
                    <span>${price_min:,.0f}</span>
                    <span style="color: var(--terracotta); font-weight: 600;">Your estimate</span>
                    <span>${price_max:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            avg_price = y.mean() * 1000
            median_price = y.median() * 1000
            diff_from_avg = price - avg_price
            diff_pct = (diff_from_avg / avg_price) * 100
            direction = "above" if diff_from_avg > 0 else "below"
            diff_color = "var(--sage)" if diff_from_avg > 0 else "var(--terracotta)"

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-item">
                    <div class="stat-value" style="color: {diff_color};">{diff_pct:+.1f}%</div>
                    <div class="stat-label">{direction} avg</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${avg_price:,.0f}</div>
                    <div class="stat-label">Dataset Avg</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${median_price:,.0f}</div>
                    <div class="stat-label">Dataset Median</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Samples</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="card">
            <p class="section-label" style="color: var(--plum);">Feature Importance</p>
        </div>
        """, unsafe_allow_html=True)

        for idx, (_, row) in enumerate(importance_df.iterrows()):
            feat = row['Feature']
            imp = row['Importance']
            pct_imp = imp / importance_df['Importance'].max() * 100
            bar_color = BAR_COLORS[idx % len(BAR_COLORS)]
            info = FEATURE_INFO.get(feat, {"label": feat})

            st.markdown(f"""
            <div class="feat-bar-container">
                <div class="feat-bar-label">
                    <span class="feat-bar-name">{info['label']}</span>
                    <span class="feat-bar-val">{imp:.3f}</span>
                </div>
                <div class="feat-bar-track">
                    <div class="feat-bar-fill" style="width: {pct_imp:.1f}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────── TAB 2: INSIGHTS ────────────
with tab_insights:
    st.markdown("""
    <div class="card">
        <p class="section-label" style="color: var(--sage);">Dataset Overview</p>
        <p style="font-size: 0.88rem; color: #8E8E93;">Summary statistics of the Boston Housing dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{len(df)}")
    col2.metric("Features", f"{len(feature_order)}")
    col3.metric("Avg Price", f"${y.mean() * 1000:,.0f}")
    col4.metric("Std Dev", f"${y.std() * 1000:,.0f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label" style="color: var(--terracotta);">Price Distribution</p>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"MEDV": y}), color="#C4553A")
    with c2:
        st.markdown('<p class="section-label" style="color: var(--sage);">Rooms vs Price</p>', unsafe_allow_html=True)
        scatter_df = pd.DataFrame({"Rooms": df["RM"], "Price": df["MEDV"]}).dropna()
        st.scatter_chart(scatter_df, x="Rooms", y="Price", color="#3D7A5F")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="color: var(--plum);">Correlation with Price</p>', unsafe_allow_html=True)
    corr = df.corr(numeric_only=True)["MEDV"].drop("MEDV").sort_values(key=abs, ascending=False)

    for feat_name, val in corr.items():
        info = FEATURE_INFO.get(feat_name, {"label": feat_name})
        bar_pct = abs(val) * 100
        bar_color = "var(--sage)" if val > 0 else "var(--terracotta)"
        sign = "+" if val > 0 else "-"

        st.markdown(f"""
        <div class="feat-bar-container">
            <div class="feat-bar-label">
                <span class="feat-bar-name">{info['label']}</span>
                <span class="feat-bar-val" style="color: {bar_color};">{sign}{abs(val):.3f}</span>
            </div>
            <div class="feat-bar-track">
                <div class="feat-bar-fill" style="width: {bar_pct:.1f}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────── TAB 3: DATA EXPLORER ────────────
with tab_data:
    st.markdown("""
    <div class="card">
        <p class="section-label" style="color: var(--sand);">Raw Dataset</p>
        <p style="font-size: 0.88rem; color: #8E8E93;">Browse and sort through the complete dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        sort_by = st.selectbox("Sort by", options=list(df.columns), index=len(df.columns) - 1)
    with fcol2:
        sort_order = st.selectbox("Order", options=["Descending", "Ascending"])

    sorted_df = df.sort_values(sort_by, ascending=(sort_order == "Ascending"))
    st.dataframe(sorted_df, use_container_width=True, height=480)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label" style="color: var(--text-mid);">Descriptive Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
