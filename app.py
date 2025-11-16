import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

import matplotlib.pyplot as plt


def add_custom_style():
    # Luxury / clean UI styling
    st.markdown(
        """
        <style>
        /* Page background + width */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        body {
            background-color: #f5f4f1;
        }

        /* Title */
        h1 {
            font-size: 2.8rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase;
            color: #111827 !important;
        }

        h2, h3 {
            color: #111827 !important;
        }

        /* Tabs as pills */
        .stTabs [role="tablist"] {
            gap: 0.6rem;
        }

        .stTabs [role="tab"] {
            padding: 0.45rem 1.3rem;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background-color: #f9fafb;
            color: #374151;
            font-weight: 500;
            font-size: 0.95rem;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #111827;
            color: #f9fafb;
            border-color: #111827;
        }

        /* DataFrames in soft cards */
        .stDataFrame, .stTable {
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }

        /* Expander header bold + subtle */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
        }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
        section[data-testid="stSidebar"] * {
            color: #f9fafb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="ATELIER 8 – Circular Luxury Analytics Dashboard", layout="wide")
    add_custom_style()

    # ------------------------------------------------------------
    # HERO HEADER
    # ------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("ATELIER 8 – Circular Luxury Analytics Dashboard")
    with col2:
        st.markdown(
            """
            <div style="
                background-color:#111827;
                color:#f9fafb;
                padding:0.5rem 0.9rem;
                border-radius:999px;
                text-align:center;
                font-size:0.8rem;
                margin-top:0.6rem;
            ">
                Data Lab · Dubai · Sneaker & Bag Care
            </div>
            """,
            unsafe_allow_html=True,
        )

    # (Removed the long explanatory paragraph here – you said you don't want it)

    # ------------------------------------------------------------
    # DATA INPUT
    # ------------------------------------------------------------
    st.sidebar.header("1. Upload or Use Sample Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset uploaded successfully!")
    else:
        st.sidebar.info("No file uploaded – using sample_data.csv bundled with the repo.")
        try:
            df = pd.read_csv("sample_data.csv")
        except FileNotFoundError:
            st.error("sample_data.csv not found. Please upload a CSV file to continue.")
            st.stop()

    # Quick peek – but tucked inside an expander so it’s not cluttering the top
    with st.expander("Raw Data Preview (first 5 rows)", expanded=False):
        st.dataframe(df.head())

    # ------------------------------------------------------------
    # COLUMN TYPES
    # ------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    with st.expander("Column Overview", expanded=False):
        st.write("**Numeric columns detected:**", numeric_cols if numeric_cols else "None")
        st.write("**Categorical columns detected:**", categorical_cols if categorical_cols else "None")

    # ------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview & Filters",
            "Customer Segments (Clustering)",
            "Adoption Prediction (Classification)",
            "Pricing Insight (Regression)",
        ]
    )

    # ------------------------------------------------------------
    # TAB 1 – OVERVIEW
    # ------------------------------------------------------------
    with tab1:
        st.header("Overview & Filters")
        if not df.empty:
            st.sidebar.header("2. Basic Filters")
            filtered_df = df.copy()

            for col in categorical_cols[:3]:
                values = ["All"] + sorted(filtered_df[col].dropna().unique().tolist())
                choice = st.sidebar.selectbox(f"Filter by {col}", values, key=f"filter_{col}")
                if choice != "All":
                    filtered_df = filtered_df[filtered_df[col] == choice]

            st.subheader("Filtered Data")
            st.dataframe(filtered_df.head())

            if numeric_cols:
                num_col = st.selectbox("Choose a numeric column to visualize", numeric_cols)
                fig, ax = plt.subplots()
                ax.hist(filtered_df[num_col].dropna(), bins=20)
                ax.set_xlabel(num_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {num_col}")
                st.pyplot(fig)
            else:
                st.info("No numeric columns found for visualization.")
        else:
            st.warning("Dataset is empty. Please upload a valid CSV.")

    # ------------------------------------------------------------
    # TAB 2 – CLUSTERING
    # ------------------------------------------------------------
    with tab2:
        st.header("Customer Segments via Clustering")

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for clustering.")
        else:
            cluster_features = st.multiselect(
                "Choose numeric features for clustering",
                numeric_cols,
                default=numeric_cols[:3],
            )

            if len(cluster_features) >= 2:
                n_clusters = st.slider("Number of clusters (segments)", 2, 8, 4)

                X = df[cluster_features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                cluster_df = X.copy()
                cluster_df["cluster"] = clusters

                st.write("### Cluster Summary (feature means by segment)")
                st.dataframe(cluster_df.groupby("cluster").mean().round(2))

                fig2, ax2 = plt.subplots()
                ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
                ax2.set_xlabel(cluster_features[0])
                ax2.set_ylabel(cluster_features[1])
                ax2.set_title("Customer Segments (First 2 features)")
                st.pyplot(fig2)
            else:
                st.info("Select at least 2 features for clustering.")

    # ------------------------------------------------------------
    # TAB 3 – CLASSIFICATION
    # ------------------------------------------------------------
    with tab3:
        st.header("Adoption Prediction – Classification")

        if categorical_cols:
            target_col = st.selectbox(
                "Choose the target column (e.g., 'adoption_intent')",
                categorical_cols,
            )

            y_raw = df[target_col].astype(str)
            y = y_raw.astype("category").cat.codes

            if len(numeric_cols) == 0:
                st.warning("No numeric features available for classification.")
            else:
                feature_cols = st.multiselect(
                    "Choose numeric features for the model",
                    numeric_cols,
                    default=numeric_cols[:3],
                )

                if feature_cols:
                    X = df[feature_cols].fillna(df[feature_cols].median())

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=y
                    )

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    st.write(f"**Accuracy:** {acc:.3f}")
                    st.text("Classification report (encoded classes):")
                    st.text(classification_report(y_test, y_pred, zero_division=0))

                    st.write("### Try a Single Prediction")
                    input_data = []
                    for col in feature_cols:
                        col_min = float(X[col].min())
                        col_max = float(X[col].max())
                        col_med = float(X[col].median())
                        val = st.number_input(
                            f"Enter value for {col}",
                            min_value=col_min,
                            max_value=col_max,
                            value=col_med,
                        )
                        input_data.append(val)

                    if st.button("Predict Adoption Intent"):
                        input_array = np.array(input_data).reshape(1, -1)
                        pred_class = model.predict(input_array)[0]
                        label_map = dict(enumerate(y_raw.astype("category").cat.categories))
                        st.success(f"Predicted class: {label_map.get(pred_class, pred_class)}")
                else:
                    st.info("Please select at least one feature column for the model.")
        else:
            st.info("No categorical columns found to use as a target.")

    # ------------------------------------------------------------
    # TAB 4 – REGRESSION
    # ------------------------------------------------------------
    with tab4:
        st.header("Pricing Insight – Regression (Willingness to Pay)")

        if len(numeric_cols) >= 2:
            target_reg = st.selectbox(
                "Choose numeric target column (e.g., 'wtp_restoration')",
                numeric_cols,
            )
            feature_reg = st.multiselect(
                "Choose numeric features to explain the target",
                [c for c in numeric_cols if c != target_reg],
                default=[c for c in numeric_cols if c != target_reg][:3],
            )

            if feature_reg:
                df_reg = df[feature_reg + [target_reg]].dropna()
                Xr = df_reg[feature_reg]
                yr = df_reg[target_reg]

                Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                    Xr, yr, test_size=0.25, random_state=42
                )

                reg = LinearRegression()
                reg.fit(Xr_train, yr_train)

                yr_pred = reg.predict(Xr_test)

                mse = mean_squared_error(yr_test, yr_pred)
                rmse = float(mse ** 0.5)
                r2 = r2_score(yr_test, yr_pred)

                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**R² score:** {r2:.3f}")

                coef_df = pd.DataFrame(
                    {
                        "feature": feature_reg,
                        "coefficient": reg.coef_,
                    }
                )
                st.write("### Coefficients (Impact on Willingness to Pay)")
                st.dataframe(coef_df)

                fig3, ax3 = plt.subplots()
                ax3.scatter(yr_test, yr_pred)
                ax3.set_xlabel("Actual")
                ax3.set_ylabel("Predicted")
                ax3.set_title("Actual vs Predicted")
                st.pyplot(fig3)
            else:
                st.info("Please select at least one feature for regression.")
        else:
            st.info("Need at least 2 numeric columns for regression.")


if __name__ == "__main__":
    main()
