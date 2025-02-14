"""Logistic Regression model implementation"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from .base_model import BaseModel
from config.model_config import LR_PARAM_GRID


class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__(random_state)
        self.pipeline = None

    def train(self):
        """Train the Logistic Regression model with hyperparameter tuning"""
        self.pipeline = Pipeline(
            [
                ("smote", SMOTE(random_state=self.random_state)),
                (
                    "logreg",
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                ),
            ]
        )

        grid_search = GridSearchCV(
            self.pipeline, param_grid=LR_PARAM_GRID, cv=5, scoring="recall", n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }
