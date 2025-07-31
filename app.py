import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title
st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")
st.title("📊 Bank Marketing Prediction Dashboard")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv", sep=';')

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocess
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)

# Show metrics
st.markdown(f"### ✅ Accuracy: `{acc * 100:.2f}%`")
with st.expander("📋 Classification Report"):
    st.code(report)

# Feature importances
importances = model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
fig = px.bar(feature_df.head(10), x='Importance', y='Feature', orientation='h',
             title="Top 10 Feature Importances", color='Importance',
             color_continuous_scale='Blues')

st.plotly_chart(fig, use_container_width=True)

st.markdown("🔁 Model: Decision Tree Classifier (max_depth=5)")

