
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.title("🎮 Gaming Customer Intelligence System (Advanced)")

# Load data
df = pd.read_csv("data.csv")

# Sidebar
menu = st.sidebar.selectbox("Navigation", [
    "Overview",
    "Descriptive Analysis",
    "Diagnostic Analysis",
    "Predictive Analysis",
    "Clustering",
    "Association Rules",
    "Prescriptive Insights",
    "Upload & Predict"
])

# Encoding
df_enc = df.copy()
le_dict = {}
for col in df_enc.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    le_dict[col] = le

# ================= OVERVIEW =================
if menu == "Overview":
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# ================= DESCRIPTIVE =================
elif menu == "Descriptive Analysis":
    st.subheader("Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(df, x="Yearly_Spend", title="Spending Distribution"))
        st.plotly_chart(px.pie(df, names="Income", title="Income Distribution"))

    with col2:
        st.plotly_chart(px.bar(df["Gaming_Frequency"].value_counts(), title="Gaming Frequency"))
        st.plotly_chart(px.box(df, x="Price_Sensitivity", y="Yearly_Spend", title="Price vs Spend"))

    st.plotly_chart(px.scatter(df, x="Aesthetic", y="Yearly_Spend", color="Gaming_Frequency", title="Aesthetic vs Spend"))

# ================= DIAGNOSTIC =================
elif menu == "Diagnostic Analysis":
    st.subheader("Diagnostic Analysis")

    corr = df_enc.corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Matrix"))

    st.plotly_chart(px.box(df, x="Interest", y="Yearly_Spend", title="Interest vs Spend"))

    st.plotly_chart(px.bar(df.groupby("Engagement")["Yearly_Spend"].mean().reset_index(),
                          x="Engagement", y="Yearly_Spend", title="Engagement Impact"))

# ================= PREDICTIVE =================
elif menu == "Predictive Analysis":
    st.subheader("Classification Model")

    X = df_enc.drop("Interest", axis=1)
    y = df_enc["Interest"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

    # ROC
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.plotly_chart(px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})"))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"))

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(importance)

# ================= CLUSTERING =================
elif menu == "Clustering":
    st.subheader("Customer Segmentation")

    kmeans = KMeans(n_clusters=3)
    df["Cluster"] = kmeans.fit_predict(df_enc.drop("Interest", axis=1))

    st.plotly_chart(px.scatter(df, x="Aesthetic", y="Yearly_Spend", color="Cluster"))

    st.write(df.groupby("Cluster").mean(numeric_only=True))

# ================= ASSOCIATION =================
elif menu == "Association Rules":
    st.subheader("Association Rule Mining")

    # dummy basket (simplified)
    basket = pd.get_dummies(df[["Gaming_Frequency","Price_Sensitivity"]])
    frequent = apriori(basket, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1)

    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# ================= PRESCRIPTIVE =================
elif menu == "Prescriptive Insights":
    st.subheader("Business Recommendations")

    st.markdown("### 🎯 Target High Value Users")
    st.write("Focus on users with high engagement + low price sensitivity")

    st.markdown("### 📦 Bundle Strategy")
    st.write("Bundle high aesthetic + high spend users")

    st.markdown("### 💸 Pricing Strategy")
    st.write("Use tiered pricing based on income clusters")

# ================= UPLOAD =================
elif menu == "Upload & Predict":
    st.subheader("Upload New Customers")

    file = st.file_uploader("Upload CSV")

    if file:
        new = pd.read_csv(file)
        for col in new.select_dtypes(include='object').columns:
            if col in le_dict:
                new[col] = le_dict[col].transform(new[col])

        preds = model.predict(new)
        st.write(preds)
