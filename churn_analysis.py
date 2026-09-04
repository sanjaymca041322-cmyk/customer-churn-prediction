# ============================================================
# CUSTOMER CHURN ANALYSIS & MACHINE LEARNING PROJECT
# ============================================================
#
# Project:
# Customer Churn Prediction
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. Tuned Random Forest
#
# Additional Analysis:
# - Dataset exploration
# - Missing value analysis
# - Statistical analysis
# - Churn analysis
# - Feature importance
# - ROC-AUC
# - Customer churn probability
# - Customer risk classification
# - Model comparison
# - Visualizations
# - Business insights
# - New customer prediction
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics import make_scorer


warnings.filterwarnings("ignore")


# ============================================================
# 2. FILE PATH
# ============================================================

DATA_FILE = "data/customer_churn.csv"

OUTPUT_DIRECTORY = "data"

PREDICTION_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "churn_predictions.csv"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("DATASET EXPLORATION")
print("=" * 60)

try:

    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:

    print("\nERROR:")
    print(f"Dataset not found at: {DATA_FILE}")
    print("\nPlease make sure your CSV file exists at:")
    print("Customer_Churn_Project/data/customer_churn.csv")

    raise


# ============================================================
# 4. BASIC DATA EXPLORATION
# ============================================================

print("\nFirst 5 Records:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nColumn Names:")
print(df.columns)


print("\nDataset Information:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Numerical columns
numeric_columns = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalSpend",
    "SupportCalls"
]


# Categorical columns
categorical_columns = [
    "Gender",
    "Contract",
    "Churn"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(
            df[column].median()
        )


for column in categorical_columns:

    if column in df.columns:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary:")

print(
    df[
        [
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalSpend",
            "SupportCalls"
        ]
    ].describe()
)


# ============================================================
# 7. CUSTOMER CHURN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN ANALYSIS")
print("=" * 60)


# Churn distribution

churn_distribution = df["Churn"].value_counts()

print("\nChurn Distribution:")
print(churn_distribution)


# Churn percentage

churn_percentage = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nChurn Percentage:")
print(churn_percentage)


# ============================================================
# 8. CONTRACT TYPE VS CHURN
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("\nContract Type vs Churn:")
print(contract_churn)


# Contract churn percentage

contract_churn_percentage = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
).mul(100).round(2)


print("\nContract Type Churn Percentage:")
print(contract_churn_percentage)


# ============================================================
# 9. SUPPORT CALLS VS CHURN
# ============================================================

support_calls_churn = (
    df.groupby("Churn")["SupportCalls"]
    .mean()
)

print("\nAverage Support Calls by Churn Status:")
print(support_calls_churn)


# ============================================================
# 10. AVERAGE NUMERICAL VALUES BY CHURN
# ============================================================

average_values = (
    df.groupby("Churn")[
        [
            "Age",
            "Tenure",
            "MonthlyCharges",
            "TotalSpend",
            "SupportCalls"
        ]
    ].mean()
)

print("\nAverage Numerical Values by Churn Status:")
print(average_values)


# ============================================================
# 11. CREATE NUMERIC CHURN COLUMN
# ============================================================

df["ChurnNumeric"] = (
    df["Churn"]
    .map({
        "No": 0,
        "Yes": 1
    })
)


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalSpend",
    "SupportCalls",
    "ChurnNumeric"
]

correlation = (
    df[correlation_columns]
    .corr()["ChurnNumeric"]
    .sort_values(
        ascending=False
    )
)

print("\nCorrelation with Churn:")
print(correlation)


# ============================================================
# 13. MACHINE LEARNING DATA PREPARATION
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING DATA PREPARATION")
print("=" * 60)


features = [
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "TotalSpend",
    "Contract",
    "SupportCalls"
]


X = df[features]

y = df["Churn"]


print("\nColumns used for Machine Learning:")
print(df.columns)


print("\nFeatures (X):")
print(X.columns)


print("\nTarget (y):")
print(y.head())


# ============================================================
# 14. ENCODE CATEGORICAL FEATURES
# ============================================================

X_encoded = pd.get_dummies(
    X,
    columns=[
        "Gender",
        "Contract"
    ],
    drop_first=True
)


print("\nEncoded Features:")
print(X_encoded.head())


print("\nEncoded Columns:")
print(X_encoded.columns)


# ============================================================
# 15. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data shape:")
print(X_train.shape)


print("\nTesting data shape:")
print(X_test.shape)


print("\nTraining target distribution:")
print(y_train.value_counts())


print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 16. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 17. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


print("\nLogistic Regression model trained successfully!")


logistic_predictions = (
    logistic_model.predict(
        X_test_scaled
    )
)


logistic_probabilities = (
    logistic_model.predict_proba(
        X_test_scaled
    )[:, 1]
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print("\nActual Churn:")
print(y_test.head(10).to_numpy())


print("\nPredicted Churn:")
print(logistic_predictions[:10])


print("\nLogistic Regression Accuracy:")
print(f"{logistic_accuracy * 100:.2f}%")


print("\nLogistic Regression Classification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)


logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=["No", "Yes"]
)


print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)


# ============================================================
# 18. DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)


decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


decision_tree.fit(
    X_train,
    y_train
)


print("\nDecision Tree model trained successfully!")


decision_tree_predictions = (
    decision_tree.predict(
        X_test
    )
)


decision_tree_probabilities = (
    decision_tree.predict_proba(
        X_test
    )[:, 1]
)


decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_predictions
)


print("\nDecision Tree Accuracy:")
print(
    f"{decision_tree_accuracy * 100:.2f}%"
)


print("\nDecision Tree Classification Report:")

print(
    classification_report(
        y_test,
        decision_tree_predictions
    )
)


decision_tree_cm = confusion_matrix(
    y_test,
    decision_tree_predictions,
    labels=["No", "Yes"]
)


print("\nDecision Tree Confusion Matrix:")
print(decision_tree_cm)


# ============================================================
# 19. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)


random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


random_forest.fit(
    X_train,
    y_train
)


print("\nRandom Forest model trained successfully!")


random_forest_predictions = (
    random_forest.predict(
        X_test
    )
)


random_forest_probabilities = (
    random_forest.predict_proba(
        X_test
    )[:, 1]
)


random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)


print("\nRandom Forest Accuracy:")
print(
    f"{random_forest_accuracy * 100:.2f}%"
)


print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        random_forest_predictions
    )
)


random_forest_cm = confusion_matrix(
    y_test,
    random_forest_predictions,
    labels=["No", "Yes"]
)


print("\nRandom Forest Confusion Matrix:")
print(random_forest_cm)


# ============================================================
# 20. INITIAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("INITIAL MODEL COMPARISON")
print("=" * 60)


print(
    f"Logistic Regression : "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Decision Tree       : "
    f"{decision_tree_accuracy * 100:.2f}%"
)

print(
    f"Random Forest       : "
    f"{random_forest_accuracy * 100:.2f}%"
)


# ============================================================
# 21. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)


# IMPORTANT:
# Your target labels are "No" and "Yes".
#
# Therefore we explicitly specify:
# pos_label="Yes"
#
# This fixes:
# ValueError: pos_label=1 is not a valid label


f1_yes_scorer = make_scorer(
    f1_score,
    pos_label="Yes"
)


parameter_grid = {

    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        None,
        5,
        10
    ],

    "min_samples_split": [
        2,
        5
    ]
}


cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=parameter_grid,

    scoring=f1_yes_scorer,

    cv=cv_strategy,

    n_jobs=-1,

    return_train_score=False
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest Random Forest Parameters:")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation F1 Score:")

print(
    f"{grid_search.best_score_ * 100:.2f}%"
)


# ============================================================
# 22. TUNED RANDOM FOREST
# ============================================================

tuned_random_forest = (
    grid_search.best_estimator_
)


tuned_predictions = (
    tuned_random_forest.predict(
        X_test
    )
)


tuned_probabilities = (
    tuned_random_forest.predict_proba(
        X_test
    )[:, 1]
)


tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)


print("\nTuned Random Forest Accuracy:")

print(
    f"{tuned_accuracy * 100:.2f}%"
)


print("\nTuned Random Forest Classification Report:")

print(
    classification_report(
        y_test,
        tuned_predictions
    )
)


tuned_cm = confusion_matrix(
    y_test,
    tuned_predictions,
    labels=["No", "Yes"]
)


print("\nTuned Random Forest Confusion Matrix:")

print(tuned_cm)


# ============================================================
# 23. FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)


model_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],

    "Accuracy": [

        logistic_accuracy,

        decision_tree_accuracy,

        random_forest_accuracy,

        tuned_accuracy
    ],

    "F1_Yes": [

        f1_score(
            y_test,
            logistic_predictions,
            pos_label="Yes"
        ),

        f1_score(
            y_test,
            decision_tree_predictions,
            pos_label="Yes"
        ),

        f1_score(
            y_test,
            random_forest_predictions,
            pos_label="Yes"
        ),

        f1_score(
            y_test,
            tuned_predictions,
            pos_label="Yes"
        )
    ]
})


model_results["Accuracy"] = (
    model_results["Accuracy"] * 100
)

model_results["F1_Yes"] = (
    model_results["F1_Yes"] * 100
)


print(model_results.to_string(index=False))


# ============================================================
# 24. DETERMINE BEST MODEL
# ============================================================

best_model_row = (
    model_results
    .sort_values(
        by="F1_Yes",
        ascending=False
    )
    .iloc[0]
)


best_model_name = (
    best_model_row["Model"]
)

best_model_accuracy = (
    best_model_row["Accuracy"]
)

best_model_f1 = (
    best_model_row["F1_Yes"]
)


print("\nBest Model Based on Churn F1 Score:")

print(best_model_name)


print("\nBest Model Accuracy:")

print(
    f"{best_model_accuracy:.2f}%"
)


print("\nBest Model Churn F1 Score:")

print(
    f"{best_model_f1:.2f}%"
)


# ============================================================
# 25. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)


feature_importance = pd.DataFrame({

    "Feature": X_encoded.columns,

    "Importance":
        random_forest.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nFeature Importance:")

print(feature_importance)


# ============================================================
# 26. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION FEATURE COEFFICIENTS")
print("=" * 60)


logistic_coefficients = pd.DataFrame({

    "Feature":
        X_encoded.columns,

    "Coefficient":
        logistic_model.coef_[0]
})


logistic_coefficients["AbsoluteCoefficient"] = (
    logistic_coefficients["Coefficient"]
    .abs()
)


logistic_coefficients = (
    logistic_coefficients
    .sort_values(
        by="AbsoluteCoefficient",
        ascending=False
    )
)


print(
    logistic_coefficients[
        [
            "Feature",
            "Coefficient"
        ]
    ]
)


# ============================================================
# 27. ROC-AUC ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ROC-AUC ANALYSIS")
print("=" * 60)


logistic_auc = roc_auc_score(
    y_test.map({
        "No": 0,
        "Yes": 1
    }),
    logistic_probabilities
)


random_forest_auc = roc_auc_score(
    y_test.map({
        "No": 0,
        "Yes": 1
    }),
    random_forest_probabilities
)


tuned_random_forest_auc = roc_auc_score(
    y_test.map({
        "No": 0,
        "Yes": 1
    }),
    tuned_probabilities
)


decision_tree_auc = roc_auc_score(
    y_test.map({
        "No": 0,
        "Yes": 1
    }),
    decision_tree_probabilities
)


print("\nLogistic Regression ROC-AUC Score:")

print(
    f"{logistic_auc * 100:.2f}%"
)


print("\nDecision Tree ROC-AUC Score:")

print(
    f"{decision_tree_auc * 100:.2f}%"
)


print("\nRandom Forest ROC-AUC Score:")

print(
    f"{random_forest_auc * 100:.2f}%"
)


print("\nTuned Random Forest ROC-AUC Score:")

print(
    f"{tuned_random_forest_auc * 100:.2f}%"
)


# ============================================================
# 28. ROC CURVE VISUALIZATION
# ============================================================

y_test_numeric = y_test.map({
    "No": 0,
    "Yes": 1
})


logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test_numeric,
    logistic_probabilities
)


decision_tree_fpr, decision_tree_tpr, _ = roc_curve(
    y_test_numeric,
    decision_tree_probabilities
)


rf_fpr, rf_tpr, _ = roc_curve(
    y_test_numeric,
    random_forest_probabilities
)


tuned_rf_fpr, tuned_rf_tpr, _ = roc_curve(
    y_test_numeric,
    tuned_probabilities
)


plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression AUC={logistic_auc:.2f}"
)

plt.plot(
    decision_tree_fpr,
    decision_tree_tpr,
    label=f"Decision Tree AUC={decision_tree_auc:.2f}"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest AUC={random_forest_auc:.2f}"
)

plt.plot(
    tuned_rf_fpr,
    tuned_rf_tpr,
    label=f"Tuned RF AUC={tuned_random_forest_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 29. CUSTOMER CHURN PROBABILITY
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN PROBABILITY")
print("=" * 60)


# We use Logistic Regression because it has:
#
# - Best ROC-AUC
# - Good interpretability
# - Probability output
#
# Probability of "Yes" is column 1.


all_customer_encoded = X_encoded


all_customer_scaled = scaler.transform(
    all_customer_encoded
)


all_customer_probabilities = (
    logistic_model
    .predict_proba(
        all_customer_scaled
    )[:, 1]
)


customer_predictions = np.where(
    all_customer_probabilities >= 0.50,
    "Yes",
    "No"
)


prediction_df = df[
    [
        "CustomerID",
        "Age",
        "Gender",
        "Tenure",
        "MonthlyCharges",
        "TotalSpend",
        "Contract",
        "SupportCalls",
        "Churn"
    ]
].copy()


prediction_df[
    "ChurnProbability"
] = (
    all_customer_probabilities * 100
)


prediction_df[
    "PredictedChurn"
] = customer_predictions


# ============================================================
# 30. CUSTOMER RISK LEVEL
# ============================================================

def assign_risk_level(probability):

    if probability >= 70:

        return "High"

    elif probability >= 50:

        return "Medium"

    else:

        return "Low"


prediction_df[
    "RiskLevel"
] = prediction_df[
    "ChurnProbability"
].apply(
    assign_risk_level
)


prediction_df[
    "ChurnProbability"
] = prediction_df[
    "ChurnProbability"
].round(2)


# ============================================================
# 31. TOP HIGH-RISK CUSTOMERS
# ============================================================

top_risk_customers = (
    prediction_df
    .sort_values(
        by="ChurnProbability",
        ascending=False
    )
    .head(20)
)


print("\nTop 20 High-Risk Customers:")

print(
    top_risk_customers[
        [
            "CustomerID",
            "Contract",
            "SupportCalls",
            "ChurnProbability",
            "PredictedChurn",
            "RiskLevel"
        ]
    ].to_string(index=False)
)


# ============================================================
# 32. RISK DISTRIBUTION
# ============================================================

risk_distribution = (
    prediction_df[
        "RiskLevel"
    ].value_counts()
)


print("\nCustomer Risk Distribution:")

print(risk_distribution)


# ============================================================
# 33. SAVE CUSTOMER PREDICTIONS
# ============================================================

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


print("\nCustomer churn predictions saved to:")

print(PREDICTION_FILE)


# ============================================================
# 34. CHURN DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 35. CONTRACT VS CHURN VISUALIZATION
# ============================================================

contract_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Contract Type vs Churn"
)

plt.xlabel(
    "Contract Type"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 36. SUPPORT CALLS VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="SupportCalls",
    by="Churn"
)

plt.title(
    "Support Calls by Churn Status"
)

plt.suptitle("")

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Support Calls"
)

plt.tight_layout()

plt.show()


# ============================================================
# 37. MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title(
    "Monthly Charges by Churn Status"
)

plt.suptitle("")

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Monthly Charges"
)

plt.tight_layout()

plt.show()


# ============================================================
# 38. TOP FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = (
    feature_importance
    .head(8)
    .sort_values(
        by="Importance"
    )
)


plt.figure(figsize=(8, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# ============================================================
# 39. MODEL ACCURACY VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    model_results["Model"],
    model_results["Accuracy"]
)

plt.title(
    "Model Accuracy Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.show()


# ============================================================
# 40. RISK LEVEL VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

risk_distribution.plot(
    kind="bar"
)

plt.title(
    "Customer Risk Distribution"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 41. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)


monthly_churn_rate = (
    contract_churn_percentage.loc[
        "Monthly",
        "Yes"
    ]
    if "Monthly" in contract_churn_percentage.index
    and "Yes" in contract_churn_percentage.columns
    else None
)


yearly_churn_rate = (
    contract_churn_percentage.loc[
        "Yearly",
        "Yes"
    ]
    if "Yearly" in contract_churn_percentage.index
    and "Yes" in contract_churn_percentage.columns
    else None
)


support_no = (
    support_calls_churn.get(
        "No",
        np.nan
    )
)


support_yes = (
    support_calls_churn.get(
        "Yes",
        np.nan
    )
)


print("\n1. Overall Churn:")

print(
    f"Overall customer churn rate is "
    f"{churn_percentage.get('Yes', 0):.2f}%."
)


print("\n2. Contract Analysis:")

if monthly_churn_rate is not None:

    print(
        f"Monthly contract churn rate: "
        f"{monthly_churn_rate:.2f}%."
    )

if yearly_churn_rate is not None:

    print(
        f"Yearly contract churn rate: "
        f"{yearly_churn_rate:.2f}%."
    )


print("\n3. Support Calls:")

print(
    f"Average support calls for non-churned customers: "
    f"{support_no:.2f}"
)

print(
    f"Average support calls for churned customers: "
    f"{support_yes:.2f}"
)


print("\n4. Important Features:")

for _, row in feature_importance.head(5).iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


print("\n5. Customer Risk:")

print(
    f"High-risk customers: "
    f"{risk_distribution.get('High', 0)}"
)

print(
    f"Medium-risk customers: "
    f"{risk_distribution.get('Medium', 0)}"
)

print(
    f"Low-risk customers: "
    f"{risk_distribution.get('Low', 0)}"
)


# ============================================================
# 42. NEW CUSTOMER PREDICTION FUNCTION
# ============================================================

def predict_customer_churn(
    age,
    gender,
    tenure,
    monthly_charges,
    total_spend,
    contract,
    support_calls
):

    new_customer = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Tenure": [tenure],

        "MonthlyCharges": [
            monthly_charges
        ],

        "TotalSpend": [
            total_spend
        ],

        "Contract": [contract],

        "SupportCalls": [
            support_calls
        ]
    })


    new_customer_encoded = pd.get_dummies(
        new_customer,
        columns=[
            "Gender",
            "Contract"
        ],
        drop_first=True
    )


    # Make sure new customer has exactly
    # the same columns as training data

    new_customer_encoded = (
        new_customer_encoded
        .reindex(
            columns=X_encoded.columns,
            fill_value=0
        )
    )


    new_customer_scaled = (
        scaler.transform(
            new_customer_encoded
        )
    )


    probability = (
        logistic_model
        .predict_proba(
            new_customer_scaled
        )[0][1]
    )


    prediction = (
        "Yes"
        if probability >= 0.50
        else "No"
    )


    risk = assign_risk_level(
        probability * 100
    )


    print("\n" + "=" * 60)
    print("NEW CUSTOMER CHURN PREDICTION")
    print("=" * 60)


    print(
        f"\nAge: {age}"
    )

    print(
        f"Gender: {gender}"
    )

    print(
        f"Tenure: {tenure}"
    )

    print(
        f"Monthly Charges: {monthly_charges}"
    )

    print(
        f"Total Spend: {total_spend}"
    )

    print(
        f"Contract: {contract}"
    )

    print(
        f"Support Calls: {support_calls}"
    )


    print(
        f"\nChurn Probability: "
        f"{probability * 100:.2f}%"
    )


    print(
        f"Predicted Churn: "
        f"{prediction}"
    )


    print(
        f"Risk Level: "
        f"{risk}"
    )


    return {
        "ChurnProbability":
            round(probability * 100, 2),

        "PredictedChurn":
            prediction,

        "RiskLevel":
            risk
    }


# ============================================================
# 43. EXAMPLE NEW CUSTOMER PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("EXAMPLE NEW CUSTOMER")
print("=" * 60)


example_prediction = predict_customer_churn(

    age=45,

    gender="Male",

    tenure=12,

    monthly_charges=2500,

    total_spend=30000,

    contract="Monthly",

    support_calls=8
)


# ============================================================
# 44. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)


total_customers = len(df)


churned_customers = (
    df["Churn"]
    .eq("Yes")
    .sum()
)


non_churned_customers = (
    df["Churn"]
    .eq("No")
    .sum()
)


overall_churn_rate = (
    churned_customers /
    total_customers *
    100
)


print(
    f"Total Customers        : "
    f"{total_customers}"
)


print(
    f"Churned Customers      : "
    f"{churned_customers}"
)


print(
    f"Non-Churned Customers  : "
    f"{non_churned_customers}"
)


print(
    f"Overall Churn Rate     : "
    f"{overall_churn_rate:.2f}%"
)


print(
    f"Best Model             : "
    f"{best_model_name}"
)


print(
    f"Best Model Accuracy    : "
    f"{best_model_accuracy:.2f}%"
)


print(
    f"Best Model Churn F1    : "
    f"{best_model_f1:.2f}%"
)


print(
    f"Logistic ROC-AUC       : "
    f"{logistic_auc * 100:.2f}%"
)


print("\nTop Features:")

print(
    feature_importance
    .head(5)
    .to_string(index=False)
)


print("\nPrediction File:")

print(
    PREDICTION_FILE
)


# ============================================================
# 45. PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)

print(
    "CUSTOMER CHURN PROJECT COMPLETED"
)

print("=" * 60)
