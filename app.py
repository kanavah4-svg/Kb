# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# ----------------------- THEME SETUP -----------------------
st.set_page_config(
    page_title="ATELIER 8 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beige theme and fixed team section
st.markdown("""
    <style>
        body {
            background-color: #f6f1e9;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        .team-box {
            background-color: #e7dfd8;
            border-radius: 8px;
            padding: 10px;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR / TEAM --------------------
st.sidebar.title("👤 Team Members")
st.sidebar.markdown(
    """
    <div class='team-box'>
    <b>Kanav</b><br>
    <b>Omkar</b><br>
    <b>Jigyasa</b><br>
    <b>Hardik</b><br>
    <b>Harshal</b><br>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")  # Replace with your dataset path

df = load_data()

# ------------------ PLOTS ------------------
def pastel_palette():
    return sns.color_palette("pastel")

def show_overview():
    st.header("Overview & Filters")

    st.subheader("Filtered Data")
    st.dataframe(df.head())

    numeric_column = st.selectbox("Choose a numeric column to visualize", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[numeric_column], kde=True, ax=ax, color=pastel_palette()[0])
    ax.set_title(f"Distribution of {numeric_column}")
    st.pyplot(fig)

    st.subheader("Adoption Intent by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="adoption_intent", hue="gender", palette=pastel_palette(), ax=ax2)
    ax2.set_title("Adoption Intent by Gender")
    st.pyplot(fig2)

    st.subheader("Brand Preference Distribution")
    brand_counts = df['brand_preference'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(brand_counts, labels=brand_counts.index, autopct='%1.1f%%', colors=pastel_palette(), startangle=90)
    ax3.axis('equal')
    st.pyplot(fig3)


def show_clustering():
    st.header("Customer Segments (Clustering)")

    st.markdown("""
    <details>
    <summary><b>What is KMeans Clustering?</b></summary>
    <div>KMeans divides the dataset into groups (clusters) based on similarity. It's useful to identify natural segments in customer data.</div>
    </details>
    """, unsafe_allow_html=True)

    X = df.select_dtypes(include=np.number)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    df['cluster'] = kmeans.labels_

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="wtp_restoration", y="wtp_authentication", hue="cluster", palette=pastel_palette(), ax=ax)
    ax.set_title("Customer Segments")
    st.pyplot(fig)


def show_classification():
    st.header("Adoption Prediction (Classification)")

    st.markdown("""
    <details>
    <summary><b>What is Logistic Regression?</b></summary>
    <div>Logistic regression is used for binary or multi-class classification. It estimates the probability that a data point belongs to a certain class.</div>
    </details>
    """, unsafe_allow_html=True)

    X = df.select_dtypes(include=np.number)
    y = df['adoption_intent'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def show_regression():
    st.header("Pricing Insight (Regression)")

    st.markdown("""
    <details>
    <summary><b>What is Linear Regression?</b></summary>
    <div>Linear regression helps estimate the relationship between variables. Here, it predicts willingness to pay for services based on user features.</div>
    </details>
    """, unsafe_allow_html=True)

    X = df[["income_level", "num_luxury_items", "sustainability_score"]]
    y = df["wtp_restoration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color=pastel_palette()[1])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. Predicted (Restoration WTP)")
    st.pyplot(fig)

    st.markdown(f"**RMSE**: {mean_squared_error(y_test, y_pred, squared=True) ** 0.5:.2f}")
    st.markdown(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")

# ----------------- MAIN TABS -----------------
tabs = st.tabs([
    "Overview & Filters",
    "Customer Segments (Clustering)",
    "Adoption Prediction (Classification)",
    "Pricing Insight (Regression)"
])

with tabs[0]:
    show_overview()
with tabs[1]:
    show_clustering()
with tabs[2]:
    show_classification()
with tabs[3]:
    show_regression()
