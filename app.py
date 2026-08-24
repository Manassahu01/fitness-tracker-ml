import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🫀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme — "vitals monitor" palette: coral = heart/heat, amber = energy,
# teal = baseline/neutral. Mirrors real monitor color-coding conventions
# rather than a generic dark+neon default.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0B1220",
    "panel": "#121B2E",
    "panel_border": "#1F2A44",
    "text": "#E8ECF4",
    "text_muted": "#8A93A6",
    "coral": "#FF5D5D",
    "amber": "#FFB020",
    "teal": "#2DD4BF",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Manrope:wght@400;500;700;800&display=swap');

:root {{
    --bg: {COLORS["bg"]};
    --panel: {COLORS["panel"]};
    --panel-border: {COLORS["panel_border"]};
    --text: {COLORS["text"]};
    --text-muted: {COLORS["text_muted"]};
    --coral: {COLORS["coral"]};
    --amber: {COLORS["amber"]};
    --teal: {COLORS["teal"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at 15% 0%, #14203A 0%, var(--bg) 55%) !important;
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}}

[data-testid="stSidebar"] {{
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}}

[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}

h1, h2, h3 {{
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em;
}}

.hero-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}}

.hero-sub {{
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1rem;
}}

.pulse-divider {{
    width: 100%;
    height: 36px;
    margin: 0.4rem 0 1.6rem 0;
}}

.pulse-path {{
    fill: none;
    stroke: var(--coral);
    stroke-width: 2;
    stroke-linecap: round;
    stroke-dasharray: 6 4;
    animation: pulse-move 3s linear infinite;
}}

@media (prefers-reduced-motion: reduce) {{
    .pulse-path {{ animation: none; }}
}}

@keyframes pulse-move {{
    to {{ stroke-dashoffset: -100; }}
}}

.vital-card {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    height: 100%;
}}

.vital-label {{
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}}

.vital-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
}}

.vital-bar-track {{
    width: 100%;
    height: 5px;
    background: #1F2A44;
    border-radius: 3px;
    margin-top: 0.5rem;
    overflow: hidden;
}}

.vital-bar-fill {{
    height: 100%;
    border-radius: 3px;
}}

[data-testid="stTabs"] button {{
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
}}

[data-testid="stTabs"] button[aria-selected="true"] {{
    color: var(--amber) !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--panel-border);
    border-radius: 8px;
}}

footer, #MainMenu {{visibility: hidden;}}

.app-footer {{
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--panel-border);
    color: var(--text-muted);
    font-size: 0.8rem;
    text-align: center;
}}

.app-footer a {{
    color: var(--teal);
    text-decoration: none;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def style_chart(fig, height=260):
    """Apply the app's dark theme to a Plotly figure consistently."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "family": "Manrope, sans-serif"},
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Data loading (cached separately from model training so slider moves don't
# re-read/re-process the CSVs on every rerun)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
    except FileNotFoundError:
        st.error(
            "⚠️ Could not find `calories.csv` / `exercise.csv`. "
            "Make sure both files are in the app's root directory."
        )
        st.stop()

    exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

    # Normalize gender explicitly instead of relying on pd.get_dummies() column
    # naming (which silently breaks if casing/spelling in the CSV ever changes).
    exercise_df["Gender_male"] = (
        exercise_df["Gender"].astype(str).str.strip().str.lower().eq("male").astype(int)
    )

    return exercise_df


# ---------------------------------------------------------------------------
# Model training (cached as a resource, keyed on the training data itself)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    # n_estimators=1000 / max_depth=6 / max_features=3 keeps the forest deep
    # enough to fit well while limiting overfitting on this dataset size.
    # random_state=42 makes predictions reproducible across app restarts.
    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=6,
        max_features=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate on held-out test data so accuracy isn't just an assumption.
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-title">🫀 Personal Fitness Tracker</div>
    <div class="hero-sub">
        Predicting calories burned from live vitals — Age, BMI, Duration, Heart Rate &amp; Body Temp —
        using a Random Forest regressor.
    </div>
    <svg class="pulse-divider" viewBox="0 0 400 36" preserveAspectRatio="none">
        <path class="pulse-path" d="M0,18 L60,18 L75,4 L90,32 L105,18 L400,18" />
    </svg>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar for User Inputs
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ User Input Parameters")


def user_input_features():
    col1, col2 = st.sidebar.columns(2)
    age = col1.slider("Age", 10, 100, 30)
    bmi = col2.slider("BMI", 15, 40, 20)
    duration = col1.slider("Duration (min)", 0, 35, 15)
    heart_rate = col2.slider("Heart Rate", 60, 130, 80)
    body_temp = col1.slider("Body Temperature (°C)", 36, 42, 38)
    gender = 1 if st.sidebar.radio("Gender", ["Male", "Female"]) == "Male" else 0

    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration],
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })


df = user_input_features()

# ---------------------------------------------------------------------------
# Load & prep data
# ---------------------------------------------------------------------------
exercise_df = load_data()

FEATURE_COLUMNS = ["Gender_male", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]
TARGET_COLUMN = "Calories"

exercise_train, exercise_test = train_test_split(exercise_df, test_size=0.2, random_state=1)

X_train, y_train = exercise_train[FEATURE_COLUMNS], exercise_train[TARGET_COLUMN]
X_test, y_test = exercise_test[FEATURE_COLUMNS], exercise_test[TARGET_COLUMN]

# ---------------------------------------------------------------------------
# Train (cached) + Predict
# ---------------------------------------------------------------------------
with st.spinner("🔄 Running the model on your input..."):
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    prediction = model.predict(df.reindex(columns=X_train.columns, fill_value=0))

pred_value = round(float(prediction[0]), 1)

# ---------------------------------------------------------------------------
# Tabs — replaces stacked expanders with a proper dashboard layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🧠 Model Insights", "📊 Explore Data"])

# ---- Tab 1: Prediction --------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📌 Your Selected Parameters")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("🔥 Estimated Calories Burned")
        q1, q3 = exercise_df["Calories"].quantile([0.25, 0.75])
        max_range = float(exercise_df["Calories"].max() * 1.05)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_value,
            number={"suffix": " kcal", "font": {"size": 34, "color": COLORS["amber"], "family": "JetBrains Mono, monospace"}},
            gauge={
                "axis": {"range": [0, max_range], "tickcolor": COLORS["text_muted"], "tickfont": {"color": COLORS["text_muted"]}},
                "bar": {"color": COLORS["amber"], "thickness": 0.3},
                "bgcolor": COLORS["panel"],
                "borderwidth": 1,
                "bordercolor": COLORS["panel_border"],
                "steps": [
                    {"range": [0, q1], "color": "#17233D"},
                    {"range": [q1, q3], "color": "#1F2A44"},
                    {"range": [q3, max_range], "color": "#233052"},
                ],
            },
        ))
        st.plotly_chart(style_chart(gauge, height=240), use_container_width=True)

    st.write("How your input compares to the rest of the dataset:")

    age_pct = round(sum(exercise_df['Age'] < df['Age'].values[0]) / len(exercise_df) * 100, 1)
    dur_pct = round(sum(exercise_df['Duration'] < df['Duration'].values[0]) / len(exercise_df) * 100, 1)
    hr_pct = round(sum(exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]) / len(exercise_df) * 100, 1)
    temp_pct = round(sum(exercise_df['Body_Temp'] < df['Body_Temp'].values[0]) / len(exercise_df) * 100, 1)

    vitals = [
        ("Age percentile", age_pct, "var(--teal)"),
        ("Duration percentile", dur_pct, "var(--amber)"),
        ("Heart rate percentile", hr_pct, "var(--coral)"),
        ("Body temp percentile", temp_pct, "var(--coral)"),
    ]

    cards = st.columns(4)
    for card, (label, pct, color) in zip(cards, vitals):
        card.markdown(
            f"""
            <div class="vital-card">
                <div class="vital-label">{label}</div>
                <div class="vital-value" style="color:{color}">{pct}%</div>
                <div class="vital-bar-track">
                    <div class="vital-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- Tab 2: Model Insights ----------------------------------------------
with tab2:
    st.subheader("🧪 Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Test MAE", f"{metrics['test_mae']:.2f} kcal")
    m2.metric("Test R²", f"{metrics['test_r2']:.3f}")
    st.caption(
        f"Train MAE: {metrics['train_mae']:.2f} kcal · Train R²: {metrics['train_r2']:.3f} "
        "— close train/test scores mean the model isn't overfitting."
    )

    st.subheader("🔑 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    imp_fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        marker_color=COLORS["teal"],
    ))
    imp_fig.update_layout(xaxis_title="Importance", yaxis_title=None)
    st.plotly_chart(style_chart(imp_fig, height=280), use_container_width=True)
    st.caption("Which inputs the model relies on most when predicting calories burned.")

# ---- Tab 3: Explore Data -------------------------------------------------
with tab3:
    st.subheader("📈 Where You Fall in the Distribution")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=exercise_df["Calories"], marker_color=COLORS["panel_border"], nbinsx=40, name="Dataset",
    ))
    hist_fig.add_vline(
        x=pred_value, line_color=COLORS["coral"], line_width=2,
        annotation_text="You", annotation_font_color=COLORS["coral"],
    )
    hist_fig.update_layout(showlegend=False, xaxis_title="Calories", yaxis_title="Users")
    st.plotly_chart(style_chart(hist_fig, height=260), use_container_width=True)

    st.subheader("📊 Similar Results in Dataset")
    calorie_range = [pred_value - 10, pred_value + 10]
    similar_data = exercise_df[
        (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
    ]
    sample_size = min(5, len(similar_data))
    if sample_size > 0:
        st.dataframe(similar_data.sample(sample_size), use_container_width=True)
    else:
        st.info("No similar results found in the dataset for this range.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built with scikit-learn · RandomForestRegressor &nbsp;·&nbsp;
        <a href="https://github.com/Manassahu01" target="_blank">More projects on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
