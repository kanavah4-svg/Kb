
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

import matplotlib.pyplot as plt

st.set_page_config(page_title="ATELIER 8 – Data Analytics Dashboard", layout="wide")

st.title("ATELIER 8 – Circular Luxury Analytics Dashboard")
st.write(
    "This dashboard explores survey data for ATELIER 8 – a circular luxury restoration "
    "and authentication studio for designer handbags, sneakers, and leather goods."
)

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

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# ------------------------------------------------------------------
# Helper: Identify numeric & categorical cols
# ------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

with st.expander("Column Overview", expanded=False):
    st.write("**Numeric columns detected:**", numeric_cols if numeric_cols else "None")
    st.write("**Categorical columns detected:**", categorical_cols if categorical_cols else "None")

# ------------------------------------------------------------------
# Tab layout
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Filters",
    "Customer Segments (Clustering)",
    "Adoption Prediction (Classification)",
    "Pricing Insight (Regression)",
])

# ------------------------------------------------------------------
# TAB 1 – Overview & Filters
# ------------------------------------------------------------------
with tab1:
    st.header("Overview & Filters")
    if not df.empty:
        # Simple sidebar filters for up to 3 categorical columns
        st.sidebar.header("2. Basic Filters")
        filtered_df = df.copy()
        for col in categorical_cols[:3]:
            values = ["All"] + sorted(filtered_df[col].dropna().unique().tolist())
            choice = st.sidebar.selectbox(f"Filter by {col}", values, key=f"filter_{col}")
            if choice != "All":
                filtered_df = filtered_df[filtered_df[col] == choice]

        st.write("### Filtered Data")
        st.dataframe(filtered_df.head())

        # Simple numeric column chart
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

# ------------------------------------------------------------------
# TAB 2 – Clustering
# ------------------------------------------------------------------
with tab2:
    st.header("Customer Segments via Clustering")
    st.write(
        "This section uses K-Means clustering to uncover hidden customer segments, "
        "such as investment collectors, rotation enthusiasts, conscious curators, "
        "or hype sneaker owners."
    )

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
    else:
        cluster_features = st.multiselect(
            "Choose numeric features for clustering",
            numeric_cols,
            default=numeric_cols[:3]
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

            st.write("### Cluster Summary")
            st.dataframe(cluster_df.groupby("cluster").mean().round(2))

            fig2, ax2 = plt.subplots()
            ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
            ax2.set_xlabel(cluster_features[0])
            ax2.set_ylabel(cluster_features[1])
            ax2.set_title("Customer Segments (First 2 features)")
            st.pyplot(fig2)
        else:
            st.info("Select at least 2 features for clustering.")

# ------------------------------------------------------------------
# TAB 3 – Classification
# ------------------------------------------------------------------
with tab3:
    st.header("Adoption Prediction – Classification")

    if categorical_cols:
        target_col = st.selectbox(
            "Choose the target column (e.g., 'adoption_intent')",
            categorical_cols
        )

        # Convert target to binary/label encoded
        y_raw = df[target_col].astype(str)
        y = y_raw.astype("category").cat.codes

        # Feature selection – numeric only for simplicity
        if len(numeric_cols) == 0:
            st.warning("No numeric features available for classification.")
        else:
            feature_cols = st.multiselect(
                "Choose numeric features for the model",
                numeric_cols,
                default=numeric_cols[:3]
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
                st.text(classification_report(y_test, y_pred))

                st.write("### Try a Single Prediction")
                input_data = []
                for col in feature_cols:
                    val = st.number_input(
                        f"Enter value for {col}", 
                        float(X[col].min()), 
                        float(X[col].max()), 
                        float(X[col].median())
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

# ------------------------------------------------------------------
# TAB 4 – Regression
# ------------------------------------------------------------------
with tab4:
    st.header("Pricing Insight – Regression (Willingness to Pay)")
    st.write(
        "Use regression to understand how income, number of items, sustainability scores, "
        "or rarity influence willingness to pay for restoration or authentication."
    )

    if len(numeric_cols) >= 2:
        target_reg = st.selectbox(
            "Choose numeric target column (e.g., 'wtp_restoration')",
            numeric_cols
        )
        feature_reg = st.multiselect(
            "Choose numeric features to explain the target",
            [c for c in numeric_cols if c != target_reg],
            default=[c for c in numeric_cols if c != target_reg][:3]
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
            rmse = mean_squared_error(yr_test, yr_pred, squared=False)
            r2 = r2_score(yr_test, yr_pred)

            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**R² score:** {r2:.3f}")

            coef_df = pd.DataFrame({
                "feature": feature_reg,
                "coefficient": reg.coef_
            })
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
