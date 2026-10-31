import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 55)
print("       HEART DISEASE PREDICTION USING SVM")
print("=" * 55)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("\nLoading dataset...")

data = pd.read_csv("heart_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 records:")
print(data.head())


# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------

print("\n" + "-" * 55)
print("DATA PREPROCESSING")
print("-" * 55)

# Remove duplicate records
data = data.drop_duplicates()

# Remove rows containing missing values
data = data.dropna()

print("Duplicates removed.")
print("Missing values removed.")
print("Final dataset shape:", data.shape)


# ---------------------------------------------------------
# 3. FEATURES AND TARGET
# ---------------------------------------------------------

X = data.drop("target", axis=1)
y = data["target"]

print("\nFeatures:")
print(list(X.columns))

print("\nTarget:")
print("0 = No Heart Disease")
print("1 = Heart Disease")


# ---------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "-" * 55)
print("TRAIN-TEST SPLIT")
print("-" * 55)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ---------------------------------------------------------
# 6. CREATE SVM MODELS
# ---------------------------------------------------------

models = {
    "Linear": SVC(kernel="linear", C=1.0),

    "Polynomial": SVC(
        kernel="poly",
        degree=3,
        C=1.0
    ),

    "RBF": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )
}


# ---------------------------------------------------------
# 7. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

results = {}

print("\n" + "=" * 55)
print("SVM MODEL RESULTS")
print("=" * 55)

for name, model in models.items():

    print("\nTraining", name, "SVM...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results[name] = {
        "model": model,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("\n", name, "SVM")
    print("-" * 30)
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1-Score :", round(f1 * 100, 2), "%")


# ---------------------------------------------------------
# 8. MODEL COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

comparison = pd.DataFrame({
    "Kernel": list(results.keys()),
    "Accuracy": [
        results[name]["accuracy"] * 100
        for name in results
    ],
    "Precision": [
        results[name]["precision"] * 100
        for name in results
    ],
    "Recall": [
        results[name]["recall"] * 100
        for name in results
    ],
    "F1-Score": [
        results[name]["f1"] * 100
        for name in results
    ]
})

print("\n", comparison.to_string(index=False))


# ---------------------------------------------------------
# 9. SELECT BEST MODEL
# ---------------------------------------------------------

best_kernel = max(
    results,
    key=lambda name: results[name]["accuracy"]
)

best_model = results[best_kernel]["model"]

print("\n" + "=" * 55)
print("BEST MODEL")
print("=" * 55)

print("Best SVM Kernel:", best_kernel)
print(
    "Best Accuracy:",
    round(
        results[best_kernel]["accuracy"] * 100,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------------------

best_predictions = results[best_kernel]["predictions"]

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\n" + "=" * 55)
print("CONFUSION MATRIX")
print("=" * 55)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Disease",
        "Heart Disease"
    ]
)

disp.plot()

plt.title(
    "Confusion Matrix - "
    + best_kernel
    + " SVM"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 11. KERNEL ACCURACY GRAPH
# ---------------------------------------------------------

kernels = list(results.keys())

accuracies = [
    results[name]["accuracy"] * 100
    for name in kernels
]

plt.figure(figsize=(8, 5))

plt.bar(kernels, accuracies)

plt.title("SVM Kernel Accuracy Comparison")
plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 12. NEW PATIENT PREDICTION
# ---------------------------------------------------------

print("\n" + "=" * 55)
print("NEW PATIENT PREDICTION")
print("=" * 55)

print("\nEnter patient details.")

try:

    patient_values = []

    for column in X.columns:

        value = float(
            input(f"Enter {column}: ")
        )

        patient_values.append(value)

    patient = pd.DataFrame(
        [patient_values],
        columns=X.columns
    )

    patient_scaled = scaler.transform(patient)

    prediction = best_model.predict(
        patient_scaled
    )

    print("\n" + "=" * 55)

    if prediction[0] == 1:
        print("RESULT: HEART DISEASE")
    else:
        print("RESULT: NO HEART DISEASE")

    print("=" * 55)

except ValueError:

    print("\nInvalid input.")
    print("Please enter numbers only.")


print("\nProgram completed successfully!")