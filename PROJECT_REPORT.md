# Heart Disease Prediction Project Report

## 1. Objective

This project predicts whether a patient is likely to have heart disease (`target = 1`) or not (`target = 0`) using the supplied `heart_dataset.csv` file. The Streamlit application places the SVM predictor first because SVM is the main model for this project.

## 2. Dataset

The dataset contains 1,888 rows and 13 input features plus the target:

- Continuous measurements: `age`, `trestbps`, `chol`, `thalachh`, `oldpeak`
- Binary or coded categorical measurements: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`
- Target: `target`

The target classes are approximately balanced: 977 positive and 911 negative records before cleaning.

## 3. Data Cleaning and Preprocessing

1. The CSV is loaded with pandas.
2. Infinite values are converted to missing values.
3. Duplicate rows are removed. This is important because repeated records can make evaluation look better than it really is and can cause leakage across train and test sets.
4. Numeric conversion is applied to every feature and the target. Rows with missing target or unusable feature values are removed.
5. The data is split into 80% training and 20% testing using stratification and `random_state=42`.
6. Numeric columns use median imputation followed by `StandardScaler`. Scaling is especially important for SVM, KNN, and Logistic Regression because their distance or coefficient calculations are affected by feature magnitude.
7. Categorical-coded columns use most-frequent imputation followed by one-hot encoding. This prevents numeric category codes from being treated as if their values have a meaningful continuous distance.

The same `ColumnTransformer` and test split are used for every model so the comparison is fair. The transformations are inside scikit-learn pipelines, so they are fitted only on training data.

## 4. Models

### SVM (primary)

The application uses an RBF-kernel Support Vector Classifier with `C=1.0` and `gamma="scale"`. SVM is a strong choice for this dataset because it can find a non-linear decision boundary while working well with scaled, one-hot encoded features. `probability=True` provides an estimated probability for the patient prediction screen.

### Logistic Regression

Logistic Regression is a useful linear baseline. It is fast, interpretable, and gives a good reference for whether a non-linear model is adding value.

### K-Nearest Neighbors

KNN classifies a patient from nearby training examples. It is a natural comparison for a medical dataset with scaled measurements, but it can be affected by noisy observations and the choice of `n_neighbors`.

### Random Forest

Random Forest combines many decision trees and can model feature interactions without requiring the user to specify them. It is included as a robust tree-based comparison and uses balanced class weights.

## 5. Performance

The Streamlit app calculates these metrics on the held-out test set:

- Accuracy: proportion of all correct predictions.
- Precision: proportion of predicted positive cases that are actually positive.
- Recall: proportion of actual positive cases detected by the model. This is important when missed disease cases are costly.
- F1 score: balance between precision and recall.
- ROC-AUC: ranking quality across classification thresholds; it is useful alongside accuracy for a medical classification task.

The live metric table in the app is the authoritative result because it is generated from the cleaned local file and the exact installed scikit-learn version. The app also shows a confusion matrix and a full classification report for each selected model.

For the current dataset and configuration, the verified test-set results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM (primary) | 0.793 | 0.793 | 0.780 | 0.786 | 0.871 |
| Logistic Regression | 0.777 | 0.776 | 0.763 | 0.769 | 0.855 |
| K-Nearest Neighbors | 0.802 | 0.787 | 0.814 | 0.800 | 0.882 |
| Random Forest | 0.843 | 0.870 | 0.797 | 0.832 | 0.917 |

Random Forest performs best on this split by accuracy and ROC-AUC. KNN has slightly higher recall than SVM, while SVM remains the primary model because its scaled RBF decision boundary is a strong and practical fit for the project goal.

## 6. Conclusion

SVM is the primary user-facing model because it is well suited to scaled tabular data and can represent non-linear relationships. The other three models provide useful baselines and alternatives. The model should be selected using the metric that best reflects the project goal, not accuracy alone. In a clinical setting, recall, calibration, external validation, and medical expert review would be required before real-world use.

## 7. Run the Project

From this folder, install the dependencies and start Streamlit:

```text
pip install -r requirements.txt
streamlit run "heart disease.py"
```