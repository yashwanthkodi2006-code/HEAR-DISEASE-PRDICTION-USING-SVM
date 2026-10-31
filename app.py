import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


st.title("❤️ Heart Disease Prediction Using SVM")
st.write("Machine Learning project using Support Vector Machine")


data = pd.read_csv("heart_dataset.csv")

data = data.drop_duplicates()
data = data.dropna()

X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


param_grid = {
    "C": [0.1, 1, 10, 50, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1],
    "kernel": ["rbf"]
}


grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

y_prediction = best_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction, zero_division=0)
recall = recall_score(y_test, y_prediction, zero_division=0)
f1 = f1_score(y_test, y_prediction, zero_division=0)


st.sidebar.header("Project Information")

st.sidebar.write("Algorithm: Support Vector Machine")
st.sidebar.write("Kernel: RBF")
st.sidebar.write(f"Best C: {grid_search.best_params_['C']}")
st.sidebar.write(f"Best Gamma: {grid_search.best_params_['gamma']}")
st.sidebar.write(f"Dataset Rows: {len(data)}")
st.sidebar.write(f"Dataset Columns: {len(data.columns)}")


st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Precision", f"{precision * 100:.2f}%")
col3.metric("Recall", f"{recall * 100:.2f}%")
col4.metric("F1-Score", f"{f1 * 100:.2f}%")


if accuracy >= 0.80:
    st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}% — Above 80%")
else:
    st.warning(f"⚠️ Model Accuracy: {accuracy * 100:.2f}% — Below 80%")


st.subheader("🔧 Best SVM Parameters")

st.write("**Kernel:** RBF")
st.write(f"**C:** {grid_search.best_params_['C']}")
st.write(f"**Gamma:** {grid_search.best_params_['gamma']}")


st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_prediction)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No Disease", "Heart Disease"])
ax.set_yticklabels(["No Disease", "Heart Disease"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

st.pyplot(fig)


st.header("🧑‍⚕️ Patient Details")

st.write("Use the buttons below to load fixed patient examples.")


positive_indices = [
    i for i, prediction in enumerate(y_prediction)
    if prediction == 1
]

negative_indices = [
    i for i, prediction in enumerate(y_prediction)
    if prediction == 0
]


positive_index = positive_indices[0]
negative_index = negative_indices[0]


yes_values = X_test.iloc[positive_index].tolist()
no_values = X_test.iloc[negative_index].tolist()


if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = "normal"


button1, button2 = st.columns(2)


with button1:
    if st.button(
        "❤️ YES - Heart Disease",
        use_container_width=True
    ):
        st.session_state.selected_patient = "yes"
        st.rerun()


with button2:
    if st.button(
        "💚 NO - No Heart Disease",
        use_container_width=True
    ):
        st.session_state.selected_patient = "no"
        st.rerun()


if st.session_state.selected_patient == "yes":
    selected_values = yes_values
elif st.session_state.selected_patient == "no":
    selected_values = no_values
else:
    selected_values = [
        float(X[column].median())
        for column in X.columns
    ]


patient_values = []

cols = st.columns(2)


for i, column in enumerate(X.columns):

    with cols[i % 2]:

        current_value = selected_values[i]

        if column == "age":

            value = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=int(current_value),
                step=1,
                key=f"age_{st.session_state.selected_patient}"
            )

        elif column == "sex":

            value = st.number_input(
                "Sex (0 = Female, 1 = Male)",
                min_value=0,
                max_value=1,
                value=int(current_value),
                step=1,
                key=f"sex_{st.session_state.selected_patient}"
            )

        elif column == "cp":

            value = st.number_input(
                "Chest Pain Type (cp)",
                min_value=0,
                max_value=10,
                value=int(current_value),
                step=1,
                key=f"cp_{st.session_state.selected_patient}"
            )

        elif column == "trestbps":

            value = st.number_input(
                "Resting Blood Pressure",
                min_value=50,
                max_value=250,
                value=int(current_value),
                step=1,
                key=f"trestbps_{st.session_state.selected_patient}"
            )

        elif column == "chol":

            value = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=700,
                value=int(current_value),
                step=1,
                key=f"chol_{st.session_state.selected_patient}"
            )

        elif column == "fbs":

            value = st.number_input(
                "Fasting Blood Sugar (fbs)",
                min_value=0,
                max_value=1,
                value=int(current_value),
                step=1,
                key=f"fbs_{st.session_state.selected_patient}"
            )

        elif column == "restecg":

            value = st.number_input(
                "Resting ECG (restecg)",
                min_value=0,
                max_value=10,
                value=int(current_value),
                step=1,
                key=f"restecg_{st.session_state.selected_patient}"
            )

        elif column == "thalachh":

            value = st.number_input(
                "Maximum Heart Rate (thalachh)",
                min_value=50,
                max_value=250,
                value=int(current_value),
                step=1,
                key=f"thalachh_{st.session_state.selected_patient}"
            )

        elif column == "exang":

            value = st.number_input(
                "Exercise Induced Angina (exang)",
                min_value=0,
                max_value=1,
                value=int(current_value),
                step=1,
                key=f"exang_{st.session_state.selected_patient}"
            )

        elif column == "oldpeak":

            value = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=float(current_value),
                step=0.1,
                key=f"oldpeak_{st.session_state.selected_patient}"
            )

        elif column == "slope":

            value = st.number_input(
                "Slope",
                min_value=0,
                max_value=10,
                value=int(current_value),
                step=1,
                key=f"slope_{st.session_state.selected_patient}"
            )

        elif column == "ca":

            value = st.number_input(
                "Number of Major Vessels (ca)",
                min_value=0,
                max_value=10,
                value=int(current_value),
                step=1,
                key=f"ca_{st.session_state.selected_patient}"
            )

        elif column == "thal":

            value = st.number_input(
                "Thal",
                min_value=0,
                max_value=10,
                value=int(current_value),
                step=1,
                key=f"thal_{st.session_state.selected_patient}"
            )

        else:

            value = st.number_input(
                column,
                value=float(current_value),
                key=f"{column}_{st.session_state.selected_patient}"
            )

        patient_values.append(value)


if st.session_state.selected_patient == "yes":
    st.info("❤️ YES example loaded — Heart Disease patient values")

elif st.session_state.selected_patient == "no":
    st.info("💚 NO example loaded — No Heart Disease patient values")


if st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
):

    patient = pd.DataFrame(
        [patient_values],
        columns=X.columns
    )

    patient_scaled = scaler.transform(patient)

    prediction = best_model.predict(patient_scaled)

    decision_score = best_model.decision_function(patient_scaled)[0]

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("❤️ Prediction: HEART DISEASE")

    else:

        st.success("💚 Prediction: NO HEART DISEASE")

    st.write(
        f"**SVM Decision Score:** {decision_score:.4f}"
    )


st.caption(
    "For educational demonstration only. "
    "This prediction is not a medical diagnosis."
)