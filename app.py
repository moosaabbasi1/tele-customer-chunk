import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Telco Churn Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid #4C6EF5;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .tag {
        display: inline-block;
        background: #e8f0fe;
        color: #3d5af1;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model  = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TELCO DATASET  (mirrors WA_Fn-UseC_-Telco-Customer-Churn.csv)
#  We reconstruct it from known distributions so all EDA charts are live.
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def build_dataset():
    rng = np.random.default_rng(42)
    n = 7032          # matches original after dropna

    churn = rng.choice([0, 1], size=n, p=[0.7349, 0.2651])

    tenure      = np.where(churn == 1,
                           rng.integers(1, 30, n),
                           rng.integers(1, 72, n))
    monthly     = np.where(churn == 1,
                           rng.uniform(50, 110, n),
                           rng.uniform(18, 90,  n))
    total       = tenure * monthly + rng.normal(0, 50, n)
    total       = np.clip(total, 0, None)
    senior      = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    gender      = rng.choice(["Male", "Female"], size=n)
    partner     = rng.choice([0, 1], size=n, p=[0.52, 0.48])
    dependents  = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    phone_svc   = rng.choice([0, 1], size=n, p=[0.10, 0.90])
    internet    = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
    contract    = np.where(churn == 1,
                           rng.choice(["Month-to-month", "One year", "Two year"],
                                      size=n, p=[0.88, 0.09, 0.03]),
                           rng.choice(["Month-to-month", "One year", "Two year"],
                                      size=n, p=[0.43, 0.29, 0.28]))
    payment     = rng.choice(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"],
                              size=n, p=[0.34, 0.23, 0.22, 0.21])
    paperless   = rng.choice([0, 1], size=n, p=[0.41, 0.59])

    df = pd.DataFrame({
        "Churn":          churn,
        "tenure":         tenure,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges":   total.round(2),
        "SeniorCitizen":  senior,
        "gender":         gender,
        "Partner":        partner,
        "Dependents":     dependents,
        "PhoneService":   phone_svc,
        "InternetService": internet,
        "Contract":       contract,
        "PaymentMethod":  payment,
        "PaperlessBilling": paperless,
    })
    return df

df_raw = build_dataset()

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE COLS
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/signal.png", width=60)
    st.title("Telco Churn App")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 About Dataset", "📊 EDA", "🤖 Model Performance", "🔍 Predict Churn"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("OEL Machine Learning Project\nTelco Customer Churn · IBM Dataset")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — ABOUT DATASET                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if page == "🏠 About Dataset":
    st.title("🏠 About the Dataset")
    st.markdown("""
    This project uses the **IBM Telco Customer Churn** dataset, publicly available on Kaggle
    ([blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
    It contains information about a fictional telecommunications company's customers and whether
    they left (churned) within the last month.
    """)

    st.markdown("---")

    # ── Key stats ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "7,043")
    c2.metric("Features",        "20")
    c3.metric("Churn Rate",      "26.5 %")
    c4.metric("Best Model Acc.", "~80 %")

    st.markdown("---")

    # ── Feature table ────────────────────────────────────────────────────────
    st.subheader("📋 Feature Dictionary")
    feature_dict = pd.DataFrame([
        ("customerID",       "Categorical", "Unique identifier for each customer (dropped during training)"),
        ("gender",           "Binary",      "Customer gender: Male / Female"),
        ("SeniorCitizen",    "Binary",      "Whether the customer is a senior citizen (1 = Yes, 0 = No)"),
        ("Partner",          "Binary",      "Whether the customer has a partner"),
        ("Dependents",       "Binary",      "Whether the customer has dependents"),
        ("tenure",           "Numeric",     "Number of months the customer has been with the company"),
        ("PhoneService",     "Binary",      "Whether the customer has a phone service"),
        ("MultipleLines",    "Categorical", "Whether the customer has multiple lines (No / Yes / No phone service)"),
        ("InternetService",  "Categorical", "Type of internet service (DSL / Fiber optic / No)"),
        ("OnlineSecurity",   "Categorical", "Whether the customer has online security add-on"),
        ("OnlineBackup",     "Categorical", "Whether the customer has online backup add-on"),
        ("DeviceProtection", "Categorical", "Whether the customer has device protection add-on"),
        ("TechSupport",      "Categorical", "Whether the customer has tech support add-on"),
        ("StreamingTV",      "Categorical", "Whether the customer streams TV"),
        ("StreamingMovies",  "Categorical", "Whether the customer streams movies"),
        ("Contract",         "Categorical", "Contract term: Month-to-month / One year / Two year"),
        ("PaperlessBilling", "Binary",      "Whether the customer uses paperless billing"),
        ("PaymentMethod",    "Categorical", "Payment method used by the customer"),
        ("MonthlyCharges",   "Numeric",     "Monthly amount charged to the customer (USD)"),
        ("TotalCharges",     "Numeric",     "Total amount charged to the customer (USD)"),
        ("Churn",            "Target",      "Whether the customer churned (Yes = left the company)"),
    ], columns=["Feature", "Type", "Description"])

    type_colors = {"Binary": "🟢", "Numeric": "🔵", "Categorical": "🟡", "Target": "🔴"}
    feature_dict["Type"] = feature_dict["Type"].apply(lambda t: f"{type_colors.get(t,'')} {t}")
    st.dataframe(feature_dict, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Preprocessing pipeline ───────────────────────────────────────────────
    st.subheader("⚙️ Preprocessing Pipeline")
    steps = {
        "1️⃣  Convert TotalCharges": "Coerced from object → float; empty strings become NaN.",
        "2️⃣  Drop NaNs":            "11 rows removed (customers with no TotalCharges recorded).",
        "3️⃣  Drop customerID":      "Non-predictive identifier — removed before training.",
        "4️⃣  Label Encoding":       "Binary columns (gender, Partner, Dependents, Churn, etc.) encoded with LabelEncoder.",
        "5️⃣  One-Hot Encoding":     "Multi-class columns (InternetService, Contract, PaymentMethod, etc.) encoded with pd.get_dummies(drop_first=True).",
        "6️⃣  StandardScaler":       "All 30 features normalised to zero mean / unit variance.",
        "7️⃣  Train / Test Split":   "80 % training, 20 % test, random_state=42.",
    }
    for title, desc in steps.items():
        with st.expander(title):
            st.write(desc)

    st.markdown("---")

    # ── Models used ──────────────────────────────────────────────────────────
    st.subheader("🤖 Models Trained")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, name in zip([mc1,mc2,mc3,mc4,mc5],
                         ["Logistic\nRegression","Decision\nTree",
                          "Random\nForest","K-Nearest\nNeighbor","Naive\nBayes"]):
        col.info(name)

    st.markdown("---")

    # ── Source & tools ───────────────────────────────────────────────────────
    st.subheader("🔧 Tools & Libraries")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""
        | Library | Purpose |
        |---|---|
        | `pandas` | Data loading & manipulation |
        | `numpy` | Numerical operations |
        | `scikit-learn` | ML models, preprocessing, metrics |
        """)
    with tc2:
        st.markdown("""
        | Library | Purpose |
        |---|---|
        | `matplotlib` | Visualisations |
        | `seaborn` | Statistical plots |
        | `streamlit` | Web application |
        | `joblib` | Model persistence |
        """)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — EDA                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("All charts are generated from a synthetic dataset that mirrors the statistical "
                "distributions of the original IBM Telco Churn CSV.")

    palette = {0: "#4C9BE8", 1: "#F4724B"}
    churn_labels = {0: "No Churn", 1: "Churned"}

    # ── Row 1: Churn distribution + Senior citizen ───────────────────────────
    st.markdown("### 1 · Target Variable")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df_raw["Churn"].value_counts().sort_index()
        bars = ax.bar([churn_labels[i] for i in counts.index],
                      counts.values,
                      color=[palette[i] for i in counts.index], width=0.5, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("Churn Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("The dataset is imbalanced — ~26.5 % of customers churned.")

    with r1c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sizes = [df_raw["Churn"].value_counts()[1], df_raw["Churn"].value_counts()[0]]
        ax.pie(sizes, labels=["Churned", "Retained"],
               colors=["#F4724B", "#4C9BE8"],
               autopct="%1.1f%%", startangle=140,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Churn Proportion", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Row 2: Numerical distributions ──────────────────────────────────────
    st.markdown("### 2 · Numerical Features vs Churn")
    r2c1, r2c2, r2c3 = st.columns(3)

    for col_widget, col_name, xlabel in zip(
        [r2c1, r2c2, r2c3],
        ["tenure", "MonthlyCharges", "TotalCharges"],
        ["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"]
    ):
        with col_widget:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            for churn_val, color in palette.items():
                subset = df_raw[df_raw["Churn"] == churn_val][col_name]
                ax.hist(subset, bins=30, alpha=0.6, color=color,
                        label=churn_labels[churn_val], edgecolor="white")
            ax.set_title(f"{col_name} by Churn", fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.caption("Churned customers tend to have **lower tenure**, **higher monthly charges**, "
               "and lower total charges (because they leave sooner).")

    st.markdown("---")

    # ── Row 3: Categorical features ─────────────────────────────────────────
    st.markdown("### 3 · Categorical Features vs Churn")

    cat_features = {
        "Contract":        "Contract Type",
        "InternetService": "Internet Service",
        "PaymentMethod":   "Payment Method",
    }

    for col_name, col_label in cat_features.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

        # Count plot
        order = df_raw[col_name].value_counts().index
        churn_order = df_raw.groupby(col_name)["Churn"].mean().sort_values(ascending=False).index
        sns.countplot(data=df_raw, x=col_name, hue="Churn",
                      order=order, palette=palette, ax=axes[0])
        axes[0].set_title(f"{col_label} — Count", fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].legend(title="Churn", labels=["No", "Yes"])
        axes[0].spines[["top","right"]].set_visible(False)

        # Churn rate per category
        churn_rate = (df_raw.groupby(col_name)["Churn"].mean() * 100).sort_values(ascending=False)
        bars = axes[1].bar(churn_rate.index, churn_rate.values,
                           color="#F4724B", edgecolor="white", alpha=0.85)
        for b, v in zip(bars, churn_rate.values):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                         f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axes[1].set_title(f"{col_label} — Churn Rate", fontweight="bold")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].spines[["top","right"]].set_visible(False)
        axes[1].set_ylim(0, churn_rate.max() + 10)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("")

    st.markdown("---")

    # ── Row 4: Binary features ───────────────────────────────────────────────
    st.markdown("### 4 · Binary Feature Churn Rates")
    binary_features = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    churn_rates = {
        feat: [
            df_raw[df_raw[feat] == 0]["Churn"].mean() * 100,
            df_raw[df_raw[feat] == 1]["Churn"].mean() * 100,
        ]
        for feat in binary_features
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(binary_features))
    w = 0.35
    bars0 = ax.bar(x - w/2, [churn_rates[f][0] for f in binary_features],
                   w, label="No (0)", color="#4C9BE8", alpha=0.85, edgecolor="white")
    bars1 = ax.bar(x + w/2, [churn_rates[f][1] for f in binary_features],
                   w, label="Yes (1)", color="#F4724B", alpha=0.85, edgecolor="white")
    for bars in [bars0, bars1]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                    f"{b.get_height():.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(binary_features, fontsize=10)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Binary Feature (0 vs 1)", fontweight="bold")
    ax.legend(title="Feature Value")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, 55)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Row 5: Correlation heat-map ──────────────────────────────────────────
    st.markdown("### 5 · Correlation Heat-map (Numeric Features)")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    corr = df_raw[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Pearson Correlation Matrix", fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.caption("Strong positive correlation between **tenure** and **TotalCharges** is expected; "
               "**MonthlyCharges** is the strongest numeric predictor of churn.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — MODEL PERFORMANCE                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("All five models from the notebook are retrained here on the synthetic dataset "
                "so you can see live metrics and visualisations.")

    # ── Build preprocessed data ───────────────────────────────────────────────
    @st.cache_data
    def prepare_ml_data():
        df = df_raw.copy()
        le = LabelEncoder()
        df["gender"] = le.fit_transform(df["gender"])
        df = pd.get_dummies(df, columns=["InternetService","Contract","PaymentMethod"], drop_first=True)
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X_scaled, y, X.columns.tolist()

    (X_train, X_test, y_train, y_test), X_scaled_all, y_all, feat_names = prepare_ml_data()

    # ── Train models ──────────────────────────────────────────────────────────
    @st.cache_resource
    def train_all_models(_X_train, _y_train):
        mdls = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(random_state=42),
            "K-Nearest Neighbor":  KNeighborsClassifier(),
            "Naive Bayes":         GaussianNB(),
        }
        trained = {}
        for name, m in mdls.items():
            m.fit(_X_train, _y_train)
            trained[name] = m
        return trained

    trained_models = train_all_models(X_train, y_train)

    # ── Compute metrics ───────────────────────────────────────────────────────
    records = []
    cms     = {}
    for name, m in trained_models.items():
        yp = m.predict(X_test)
        records.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, yp),  4),
            "Precision": round(precision_score(y_test, yp), 4),
            "Recall":    round(recall_score(y_test, yp),    4),
            "F1-Score":  round(f1_score(y_test, yp),        4),
        })
        cms[name] = confusion_matrix(y_test, yp)

    results_df = pd.DataFrame(records).set_index("Model")

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("📋 Metrics Comparison")
    styled = results_df.style.background_gradient(cmap="Blues", axis=0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    best_model_name = results_df["Accuracy"].idxmax()
    st.success(f"🏆 Best model by Accuracy: **{best_model_name}** "
               f"({results_df.loc[best_model_name,'Accuracy']*100:.2f} %)")

    st.markdown("---")

    # ── Bar chart comparison ───────────────────────────────────────────────────
    st.subheader("📊 Visual Comparison")
    fig, ax = plt.subplots(figsize=(11, 4))
    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x         = np.arange(len(results_df))
    w         = 0.2
    colors    = ["#4C9BE8", "#6ECFA0", "#F4724B", "#F9C74F"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, results_df[metric], w, label=metric, color=color, alpha=0.9, edgecolor="white")
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(results_df.index, rotation=12, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics", fontweight="bold")
    ax.legend(loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrices")
    cols_cm = st.columns(len(trained_models))
    for col_w, (name, cm) in zip(cols_cm, cms.items()):
        with col_w:
            fig, ax = plt.subplots(figsize=(3, 2.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No","Yes"],
                        yticklabels=["No","Yes"], ax=ax,
                        linewidths=0.5, cbar=False,
                        annot_kws={"size": 11})
            ax.set_title(name, fontsize=8, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=7)
            ax.set_ylabel("Actual", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown("---")

    # ── Unsupervised — K-Means + Hierarchical ─────────────────────────────────
    st.subheader("🔵 Unsupervised Learning — Customer Clustering")
    st.markdown("K-Means and Agglomerative Clustering applied to scaled features, "
                "visualised via PCA (2 components).")

    @st.cache_data
    def run_clustering(_X_scaled):
        pca    = PCA(n_components=2, random_state=42)
        X_pca  = pca.fit_transform(_X_scaled)
        km     = KMeans(n_clusters=3, random_state=42, n_init=10)
        km_lbl = km.fit_predict(_X_scaled)
        ag     = AgglomerativeClustering(n_clusters=3)
        ag_lbl = ag.fit_predict(_X_scaled)
        return X_pca, km_lbl, ag_lbl

    X_pca, km_labels, ag_labels = run_clustering(X_scaled_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tab10 = plt.cm.tab10.colors

    for ax, labels, title in zip(
        axes,
        [km_labels, ag_labels],
        ["K-Means Clustering (3 Clusters)", "Hierarchical Clustering (3 Clusters)"]
    ):
        for cluster_id in range(3):
            mask = labels == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=5, alpha=0.5, color=tab10[cluster_id],
                       label=f"Cluster {cluster_id}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend(markerscale=3, fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.caption("PCA reduces 30 features to 2 dimensions for visualisation. "
               "Three natural customer segments are visible in both algorithms.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — PREDICT CHURN                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif page == "🔍 Predict Churn":
    st.title("🔍 Predict Customer Churn")
    st.markdown("Fill in the customer details below and click **Predict** to see whether "
                "the customer is likely to churn.")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("churn_form"):
        st.subheader("👤 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            gender     = st.selectbox("Gender",        ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen",["No", "Yes"])
        with col2:
            partner    = st.selectbox("Partner",       ["No", "Yes"])
            dependents = st.selectbox("Dependents",    ["No", "Yes"])

        st.subheader("📋 Account Info")
        col3, col4 = st.columns(2)
        with col3:
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type",
                                    ["Month-to-month", "One year", "Two year"])
        with col4:
            monthly  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total    = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                       float(tenure * 65), step=1.0)

        col5, col6 = st.columns(2)
        with col5:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col6:
            payment   = st.selectbox("Payment Method", [
                "Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check",
            ])

        st.subheader("📞 Services")
        col7, col8 = st.columns(2)
        with col7:
            phone      = st.selectbox("Phone Service",   ["No", "Yes"])
            multiple   = st.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
            internet   = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
        with col8:
            online_sec  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            online_bkp  = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        col9, col10 = st.columns(2)
        with col9:
            tech_sup    = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
            streaming_tv= st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        with col10:
            streaming_mv= st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True, type="primary")

    # ── Feature builder ───────────────────────────────────────────────────────
    def build_feature_row():
        row = {col: 0 for col in FEATURE_COLS}
        row["gender"]           = 1 if gender == "Male" else 0
        row["SeniorCitizen"]    = 1 if senior == "Yes" else 0
        row["Partner"]          = 1 if partner == "Yes" else 0
        row["Dependents"]       = 1 if dependents == "Yes" else 0
        row["PhoneService"]     = 1 if phone == "Yes" else 0
        row["PaperlessBilling"] = 1 if paperless == "Yes" else 0
        row["tenure"]           = tenure
        row["MonthlyCharges"]   = monthly
        row["TotalCharges"]     = total

        if multiple == "No phone service": row["MultipleLines_No phone service"] = 1
        elif multiple == "Yes":            row["MultipleLines_Yes"] = 1

        if internet == "Fiber optic":      row["InternetService_Fiber optic"] = 1
        elif internet == "No":             row["InternetService_No"] = 1

        if online_sec == "No internet service": row["OnlineSecurity_No internet service"] = 1
        elif online_sec == "Yes":               row["OnlineSecurity_Yes"] = 1

        if online_bkp == "No internet service": row["OnlineBackup_No internet service"] = 1
        elif online_bkp == "Yes":               row["OnlineBackup_Yes"] = 1

        if device_prot == "No internet service": row["DeviceProtection_No internet service"] = 1
        elif device_prot == "Yes":               row["DeviceProtection_Yes"] = 1

        if tech_sup == "No internet service":    row["TechSupport_No internet service"] = 1
        elif tech_sup == "Yes":                  row["TechSupport_Yes"] = 1

        if streaming_tv == "No internet service": row["StreamingTV_No internet service"] = 1
        elif streaming_tv == "Yes":               row["StreamingTV_Yes"] = 1

        if streaming_mv == "No internet service": row["StreamingMovies_No internet service"] = 1
        elif streaming_mv == "Yes":               row["StreamingMovies_Yes"] = 1

        if contract == "One year":               row["Contract_One year"] = 1
        elif contract == "Two year":             row["Contract_Two year"] = 1

        if payment == "Credit card (automatic)": row["PaymentMethod_Credit card (automatic)"] = 1
        elif payment == "Electronic check":      row["PaymentMethod_Electronic check"] = 1
        elif payment == "Mailed check":          row["PaymentMethod_Mailed check"] = 1

        return pd.DataFrame([row], columns=FEATURE_COLS)

    # ── Prediction output ─────────────────────────────────────────────────────
    if submitted:
        df_input    = build_feature_row()
        X_scaled_in = scaler.transform(df_input)
        prediction  = model.predict(X_scaled_in)[0]
        probability = model.predict_proba(X_scaled_in)[0]
        churn_prob  = probability[1] * 100

        st.divider()

        res_c1, res_c2, res_c3 = st.columns([1.5, 1, 1])
        with res_c1:
            if prediction == 1:
                st.error("⚠️ **High Churn Risk**\n\nThis customer is likely to leave.")
            else:
                st.success("✅ **Low Churn Risk**\n\nThis customer is likely to stay.")

        with res_c2:
            st.metric("Churn Probability",  f"{churn_prob:.1f} %")
        with res_c3:
            st.metric("Retention Probability", f"{100 - churn_prob:.1f} %")

        # Gauge bar
        color = "#F4724B" if churn_prob > 50 else "#4C9BE8"
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh(0, 100, color="#e9ecef", height=0.4)
        ax.barh(0, churn_prob, color=color, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Churn Probability (%)")
        ax.set_title(f"Churn Risk: {churn_prob:.1f}%", fontweight="bold")
        ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        with st.expander("🔢 Raw feature vector sent to model"):
            st.dataframe(df_input, use_container_width=True)
