import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

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
# App Title and Description
# ---------------------------------------------------------------------------
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.markdown(
    "This app predicts the calories burned based on user parameters like `Age`, `Gender`, `BMI`, etc. "
    "Adjust the values on the sidebar and see your estimated calorie burn."
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

# Display User Parameters
col1, col2 = st.columns(2)
with col1:
    st.subheader("📌 Your Selected Parameters")
    st.dataframe(df)

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

with col2:
    st.subheader("🔥 Estimated Calories Burned")
    st.success(f"{round(prediction[0], 2)} kcal")

# ---------------------------------------------------------------------------
# Model performance (shows this isn't just a black box)
# ---------------------------------------------------------------------------
with st.expander("🧪 Model Performance"):
    m1, m2 = st.columns(2)
    m1.metric("Test MAE", f"{metrics['test_mae']:.2f} kcal")
    m2.metric("Test R²", f"{metrics['test_r2']:.3f}")
    st.caption(
        f"Train MAE: {metrics['train_mae']:.2f} kcal · Train R²: {metrics['train_r2']:.3f} "
        "— close train/test scores mean the model isn't overfitting."
    )

# ---------------------------------------------------------------------------
# Display Similar Results
# ---------------------------------------------------------------------------
with st.expander("📊 Similar Results in Dataset"):
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[
        (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
    ]
    sample_size = min(5, len(similar_data))
    if sample_size > 0:
        st.dataframe(similar_data.sample(sample_size))
    else:
        st.info("No similar results found in the dataset for this range.")

# ---------------------------------------------------------------------------
# Insights Compared to Dataset
# ---------------------------------------------------------------------------
with st.expander("📈 General Insights"):
    st.write(f"- You are older than **{round(sum(exercise_df['Age'] < df['Age'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your exercise duration is longer than **{round(sum(exercise_df['Duration'] < df['Duration'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your heart rate is higher than **{round(sum(exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your body temperature is higher than **{round(sum(exercise_df['Body_Temp'] < df['Body_Temp'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
