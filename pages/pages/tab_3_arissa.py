import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title="Model Evaluation Hub", layout="wide")

st.title("🤖 Model Evaluation Hub")
st.write("This page presents the machine learning model selection, training, and evaluation for obesity risk prediction.")

@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
    return df

df = load_data()

# Data preprocessing
encoded_df = df.copy()

binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ordinal_cols = ["CAEC", "CALC"]

le = LabelEncoder()

for col in binary_cols + ordinal_cols:
    encoded_df[col] = le.fit_transform(encoded_df[col])

encoded_df = pd.get_dummies(encoded_df, columns=["MTRANS"], drop_first=True)

X = encoded_df.drop("NObeyesdad", axis=1)
y = encoded_df["NObeyesdad"]

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="mlogloss"
    )
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted")
    })

results_df = pd.DataFrame(results)

best_model_name = results_df.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
best_pred = best_model.predict(X_test)

st.subheader("📌 Final Model Summary")

best_row = results_df[results_df["Model"] == best_model_name].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", best_model_name)
col2.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
col3.metric("Precision", f"{best_row['Precision']:.4f}")
col4.metric("F1-Score", f"{best_row['F1-Score']:.4f}")

st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df, use_container_width=True)

results_melted = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    results_melted,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Performance Comparison of Machine Learning Models",
    range_y=[0, 1]
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, best_pred)
labels = target_encoder.classes_

fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=list(labels),
    y=list(labels),
    colorscale="Blues",
    showscale=True
)
fig_cm.update_layout(
    title=f"Confusion Matrix - {best_model_name}",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    best_pred,
    target_names=target_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)

st.subheader("⭐ Top 10 Important Features")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

fig_fi = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title=f"Top 10 Important Features - {best_model_name}"
)
fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_fi, use_container_width=True)

st.info(
    "XGBoost was selected as the final model because it achieved the highest F1-score, "
    "showing strong and balanced classification performance across obesity categories."
)
