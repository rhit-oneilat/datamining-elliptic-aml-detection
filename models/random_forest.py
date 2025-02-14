"""Random Forest model implementation"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from .base_model import BaseModel
from utils.preprocessing import EnhancedSampler
from config.model_config import RF_PARAM_DIST


class RandomForestModel(BaseModel):
    def __init__(self, random_state=42):
        super().__init__(random_state)
        self.pipeline = None

    def train(self):
        """Train the Random Forest model with hyperparameter tuning"""
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "feature_selector",
                    SelectFromModel(ExtraTreesClassifier(n_estimators=50)),
                ),
                ("sampler", EnhancedSampler(sampling_strategy="auto")),
                ("classifier", RandomForestClassifier(random_state=self.random_state)),
            ]
        )

        random_search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=RF_PARAM_DIST,
            n_iter=100,
            cv=5,
            scoring="balanced_accuracy",
            random_state=self.random_state,
            n_jobs=-1,
        )

        random_search.fit(self.X_train, self.y_train)
        self.model = random_search.best_estimator_

        return {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
        }
