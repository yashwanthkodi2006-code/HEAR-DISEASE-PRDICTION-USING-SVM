from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


st.set_page_config(
	page_title="Heart Disease Prediction",
	page_icon="H",
	layout="wide",
)

DATA_PATH = Path(__file__).with_name("heart_dataset.csv")
TARGET = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalachh", "oldpeak"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data
def load_data() -> tuple[pd.DataFrame, int]:
	data = pd.read_csv(DATA_PATH)
	original_rows = len(data)
	data = data.replace([np.inf, -np.inf], np.nan)
	data = data.drop_duplicates().dropna(subset=[TARGET]).copy()
	for column in FEATURES + [TARGET]:
		data[column] = pd.to_numeric(data[column], errors="coerce")
	data = data.dropna(subset=FEATURES + [TARGET])
	data[TARGET] = data[TARGET].astype(int)
	return data, original_rows


@st.cache_resource
def train_models(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
	x = data[FEATURES]
	y = data[TARGET]
	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.2, random_state=42, stratify=y
	)

	numeric_pipeline = Pipeline(
		[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
	)
	categorical_pipeline = Pipeline(
		[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	preprocessor = ColumnTransformer(
		[
			("numeric", numeric_pipeline, NUMERIC_FEATURES),
			("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
		]
	)

	model_definitions = {
		"SVM (primary)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
		"Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
		"K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
		"Random Forest": RandomForestClassifier(
			n_estimators=300, random_state=42, class_weight="balanced"
		),
	}
	models = {}
	rows = []
	reports = {}
	for name, estimator in model_definitions.items():
		pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
		pipeline.fit(x_train, y_train)
		predictions = pipeline.predict(x_test)
		probabilities = pipeline.predict_proba(x_test)[:, 1]
		models[name] = pipeline
		rows.append(
			{
				"Model": name,
				"Accuracy": accuracy_score(y_test, predictions),
				"Precision": precision_score(y_test, predictions, zero_division=0),
				"Recall": recall_score(y_test, predictions, zero_division=0),
				"F1 Score": f1_score(y_test, predictions, zero_division=0),
				"ROC-AUC": roc_auc_score(y_test, probabilities),
			}
		)
		reports[name] = {
			"matrix": confusion_matrix(y_test, predictions),
			"text": classification_report(y_test, predictions, zero_division=0),
		}
	return models, pd.DataFrame(rows), reports


def number_input(label: str, data: pd.DataFrame, column: str) -> float:
	return st.number_input(
		label,
		min_value=float(data[column].min()),
		max_value=float(data[column].max()),
		value=float(data[column].median()),
		step=0.1,
	)


def category_input(label: str, data: pd.DataFrame, column: str) -> int:
	values = sorted(data[column].dropna().astype(int).unique().tolist())
	return int(st.selectbox(label, values, index=0))


def main() -> None:
	st.title("Heart Disease Prediction")
	st.caption("A supervised learning comparison built from the local heart disease dataset")

	try:
		data, original_rows = load_data()
	except FileNotFoundError:
		st.error(f"Could not find {DATA_PATH.name} beside this app.")
		st.stop()

	models, metrics, reports = train_models(data)

	st.header("SVM Prediction")
	st.write("Enter patient measurements below. The SVM with an RBF kernel is the primary model.")
	with st.form("prediction_form"):
		left, middle, right = st.columns(3)
		with left:
			age = number_input("Age", data, "age")
			sex = category_input("Sex (0 = female, 1 = male)", data, "sex")
			cp = category_input("Chest pain type (cp)", data, "cp")
			trestbps = number_input("Resting blood pressure", data, "trestbps")
			chol = number_input("Cholesterol", data, "chol")
		with middle:
			fbs = category_input("Fasting blood sugar > 120 (fbs)", data, "fbs")
			restecg = category_input("Resting ECG result (restecg)", data, "restecg")
			thalachh = number_input("Maximum heart rate", data, "thalachh")
			exang = category_input("Exercise-induced angina (exang)", data, "exang")
		with right:
			oldpeak = number_input("ST depression (oldpeak)", data, "oldpeak")
			slope = category_input("Slope of peak exercise ST segment", data, "slope")
			ca = category_input("Number of major vessels (ca)", data, "ca")
			thal = category_input("Thalassemia result (thal)", data, "thal")

		submitted = st.form_submit_button("Predict with SVM", type="primary", use_container_width=True)

	if submitted:
		patient = pd.DataFrame(
			[
				{
					"age": age,
					"sex": sex,
					"cp": cp,
					"trestbps": trestbps,
					"chol": chol,
					"fbs": fbs,
					"restecg": restecg,
					"thalachh": thalachh,
					"exang": exang,
					"oldpeak": oldpeak,
					"slope": slope,
					"ca": ca,
					"thal": thal,
				}
			]
		)
		svm = models["SVM (primary)"]
		prediction = int(svm.predict(patient)[0])
		probability = float(svm.predict_proba(patient)[0, 1])
		if prediction == 1:
			st.error(f"SVM result: higher likelihood of heart disease ({probability:.1%})")
		else:
			st.success(f"SVM result: lower likelihood of heart disease ({1 - probability:.1%})")
		st.caption("This is a machine-learning estimate, not a medical diagnosis.")

	st.divider()
	st.header("Model Comparison")
	st.dataframe(
		metrics.style.format({column: "{:.3f}" for column in metrics.columns if column != "Model"}),
		use_container_width=True,
		hide_index=True,
	)
	best_model = metrics.loc[metrics["ROC-AUC"].idxmax(), "Model"]
	st.info(f"Highest ROC-AUC on the held-out test set: {best_model}.")

	selected_model = st.selectbox("Inspect a model", metrics["Model"].tolist(), index=0)
	matrix = reports[selected_model]["matrix"]
	figure, axis = plt.subplots(figsize=(4, 3))
	axis.imshow(matrix, cmap="Blues")
	axis.set_title(f"{selected_model} confusion matrix")
	axis.set_xlabel("Predicted label")
	axis.set_ylabel("Actual label")
	for row in range(2):
		for column in range(2):
			axis.text(column, row, matrix[row, column], ha="center", va="center")
	st.pyplot(figure, use_container_width=False)

	with st.expander("Dataset and preprocessing details"):
		st.write(f"Rows loaded: {original_rows}; rows after cleaning and deduplication: {len(data)}.")
		st.write("Numeric features were median-imputed and standardized. Categorical codes were mode-imputed and one-hot encoded.")
		st.write("The data was split 80/20 with stratification and random_state=42. All four models used the same preprocessing and test split.")
		st.dataframe(data.head(), use_container_width=True, hide_index=True)

	with st.expander("Classification report"):
		st.code(reports[selected_model]["text"])


if __name__ == "__main__":
	main()
