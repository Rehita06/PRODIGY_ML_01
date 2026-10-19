
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT / "reports" / "linear_regression_metrics.csv"

FEATURES = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt",
]

TARGET = "SalePrice"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stHeader"] {
        background-color: #0B1120;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    .hero {
        background: linear-gradient(135deg, #172554, #123047);
        padding: 28px;
        border: 1px solid #294563;
        border-radius: 18px;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #F8FAFC;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #B6C7DC;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        color: #F1F5F9;
        font-size: 23px;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 14px;
    }

    .info-card {
        background-color: #111C30;
        border: 1px solid #27364D;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .info-label {
        color: #94A3B8;
        font-size: 13px;
    }

    .info-value {
        color: #F8FAFC;
        font-size: 25px;
        font-weight: 700;
        margin-top: 5px;
    }

    .price-card {
        background: linear-gradient(135deg, #123B35, #102C35);
        border: 1px solid #237A68;
        border-radius: 18px;
        padding: 25px;
        margin-top: 18px;
        margin-bottom: 18px;
    }

    .price-label {
        color: #A7F3D0;
        font-size: 15px;
    }

    .price-value {
        color: #6EE7B7;
        font-size: 36px;
        font-weight: 800;
        margin-top: 8px;
    }

    .small-note {
        color: #94A3B8;
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #111C30;
        border: 1px solid #27364D;
        padding: 17px;
        border-radius: 13px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #10B981;
        color: #06251D;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 43px;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #34D399;
        color: #06251D;
    }

    div[data-testid="stForm"] {
        background-color: #111827;
        border: 1px solid #27364D;
        border-radius: 15px;
        padding: 22px;
    }

    hr {
        border-color: #27364D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data(path_string):
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    return pd.read_csv(path)


@st.cache_resource
def load_model(path_string):
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {path}. "
            "Run python src/train.py first."
        )

    return joblib.load(path)


@st.cache_data
def load_metrics(path_string):
    path = Path(path_string)

    if not path.is_file():
        return None

    return pd.read_csv(path)


try:
    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
    metrics_df = load_metrics(str(METRICS_PATH))

except (FileNotFoundError, pd.errors.EmptyDataError) as error:
    st.error(f"Unable to load project files: {error}")
    st.info(
        "Check your data/raw and models folders, "
        "then run the training script if necessary."
    )
    st.stop()

except Exception as error:
    st.error(f"An unexpected loading error occurred: {error}")
    st.stop()


# Validate dataset columns
required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# Validate the saved model's expected features when available
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None and list(model_features) != FEATURES:
    st.error(
        "The saved model's feature columns do not match "
        "the prediction form. Please retrain it using "
        "the six features defined in src/train.py."
    )
    st.stop()

with st.sidebar:
    st.markdown("# 🏠 House Price Predictor")
    st.caption("MACHINE LEARNING DASHBOARD")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏡 Price Prediction",
            "📊 Data Analytics",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("### Project Information")
    st.caption("Algorithm: Linear Regression")
    st.caption("Dataset: Ames Housing")
    st.caption(f"Training records: {len(df):,}")
    st.caption("Input features: 6")

st.markdown(
    """
    <div class="hero">
        <h1>House Price Predictor</h1>
        <p>
            Explore housing data and estimate house sale prices.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

average_price = df[TARGET].mean()
median_price = df[TARGET].median()
average_area = df["GrLivArea"].mean()

if page == "🏡 Price Prediction":

    st.markdown(
        '<div class="section-title">House Price Estimator</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter the property details below to estimate its "
        "sale price using the trained Linear Regression model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Houses in Dataset", f"{len(df):,}")
    col2.metric("Average Sale Price", f"${average_price:,.0f}")
    col3.metric("Average Living Area", f"{average_area:,.0f} sq ft")

    st.markdown("### Property Details")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            st.markdown("#### 📐 Size & Rooms")

            living_area = st.number_input(
                "Living area (sq ft)",
                min_value=1,
                max_value=10000,
                value=1500,
                step=50,
                help="Above-ground living area in square feet.",
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=3,
                step=1,
            )

            bathrooms = st.number_input(
                "Full bathrooms",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
            )

        with right:
            st.markdown("#### 🏗️ Quality & Construction")

            quality = st.slider(
                "Overall quality",
                min_value=1,
                max_value=10,
                value=5,
                help="Overall material and finish quality rating.",
            )

            garage_cars = st.number_input(
                "Garage capacity (cars)",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
            )

            year_built = st.number_input(
                "Year built",
                min_value=1800,
                max_value=2026,
                value=2000,
                step=1,
            )

        submitted = st.form_submit_button(
            "✨ Estimate House Price",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame(
            [{
                "GrLivArea": living_area,
                "BedroomAbvGr": bedrooms,
                "FullBath": bathrooms,
                "OverallQual": quality,
                "GarageCars": garage_cars,
                "YearBuilt": year_built,
            }],
            columns=FEATURES,
        )

        try:
            prediction = float(model.predict(input_data)[0])

            if not np.isfinite(prediction):
                st.error("The model returned an invalid prediction.")

            elif prediction <= 0:
                st.warning(
                    f"The model returned ${prediction:,.0f}. "
                    "This is not a meaningful positive sale-price "
                    "estimate. Try different property inputs."
                )

            else:
                st.markdown(
                    f"""
                    <div class="price-card">
                        <div class="price-label">
                            ESTIMATED SALE PRICE · USD
                        </div>
                        <div class="price-value">
                            ${prediction:,.0f}
                        </div>
                        <p class="small-note">
                            Model estimate based on the six selected
                            property features.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### Your Property Summary")

                s1, s2, s3 = st.columns(3)
                s1.metric("Living Area", f"{living_area:,} sq ft")
                s2.metric("Bedrooms", str(bedrooms))
                s3.metric("Bathrooms", str(bathrooms))

                s4, s5, s6 = st.columns(3)
                s4.metric("Quality Rating", f"{quality}/10")
                s5.metric("Garage Capacity", str(garage_cars))
                s6.metric("Year Built", str(year_built))


        except Exception as error:
            st.error(f"Prediction failed: {error}")


elif page == "📊 Data Analytics":

    st.markdown(
        '<div class="section-title">Housing Data Analytics</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Explore the distribution, relationships, and summary "
        "statistics in the historical housing dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Houses", f"{len(df):,}")
    c2.metric("Average Sale Price", f"${average_price:,.0f}")
    c3.metric("Median Sale Price", f"${median_price:,.0f}")
    c4.metric("Average Living Area", f"{average_area:,.0f} sq ft")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Sale Price Distribution")

        prices = df[TARGET].dropna()

        if not prices.empty:
            histogram = np.histogram(prices, bins=25)
            histogram_df = pd.DataFrame({
                "Price Range": [
                    f"${histogram[1][i] / 1000:,.0f}k–"
                    f"${histogram[1][i + 1] / 1000:,.0f}k"
                    for i in range(len(histogram[0]))
                ],
                "Number of Houses": histogram[0],
            })

            st.bar_chart(
                histogram_df.set_index("Price Range"),
                y="Number of Houses",
            )

        st.caption("Distribution of recorded sale prices in USD.")

    with right:
        st.markdown("### Living Area vs Sale Price")

        scatter_data = df[["GrLivArea", TARGET]].dropna()

        if not scatter_data.empty:
            st.scatter_chart(
                scatter_data,
                x="GrLivArea",
                y=TARGET,
                x_label="Living Area (sq ft)",
                y_label="Sale Price (USD)",
            )

        st.caption(
            "Each point represents a house. This relationship "
            "does not by itself prove causation."
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Average Price by Quality")

        quality_data = (
            df.groupby("OverallQual", as_index=False)[TARGET]
            .mean()
            .sort_values("OverallQual")
        )

        st.bar_chart(
            quality_data.set_index("OverallQual"),
            y=TARGET,
            x_label="Overall Quality",
            y_label="Average Sale Price (USD)",
        )

    with right:
        st.markdown("### Feature Correlations")

        corr_cols = [
            "SalePrice",
            "GrLivArea",
            "OverallQual",
            "GarageCars",
            "TotalBsmtSF",
            "YearBuilt",
        ]

        available_corr_cols = [
            col for col in corr_cols if col in df.columns
        ]

        correlation = df[available_corr_cols].corr(numeric_only=True)

        st.dataframe(
            correlation.style.format("{:.2f}").background_gradient(
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
            ),
            use_container_width=True,
        )

    with st.expander("View dataset sample"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Dataset quality checks"):
        missing_values = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        missing_values = missing_values[missing_values > 0]

        st.write(f"Duplicate rows: {df.duplicated().sum():,}")

        if missing_values.empty:
            st.success("No missing values found.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(
                missing_values.rename("Missing Values").to_frame(),
                use_container_width=True,
            )

st.divider()

st.markdown(
    """
    <p class="small-note" style="text-align:center;">
        House Price Predictor · Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True,
)
