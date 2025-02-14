"""Configuration parameters for models and feature engineering"""

# Random Forest parameters
RF_PARAM_DIST = {
    "classifier__n_estimators": [100, 250, 500, 750, 1000],
    "classifier__max_depth": [5, 10, 15, 20, 25, None],
    "classifier__min_samples_split": [2, 4, 6, 8],
    "classifier__min_samples_leaf": [1, 2, 3, 4],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_features": ["sqrt", "log2", None],
}

# Logistic Regression parameters
LR_PARAM_GRID = {
    "smote__k_neighbors": [3, 5],
    "logreg__C": [0.01, 0.1, 1, 10],
    "logreg__class_weight": ["balanced", None],
}

# Feature engineering parameters
MAX_HOPS = 3
TEST_SIZE = 0.3
RANDOM_STATE = 42
