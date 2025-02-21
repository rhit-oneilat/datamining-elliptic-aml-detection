"""Feature analysis and exploration utilities"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple


class FeatureAnalyzer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.feature_importance_rf = None
        self.mutual_info_scores = None
        self.temporal_patterns = None

    def compute_feature_importance(self, n_estimators=100) -> Dict[str, float]:
        """Compute feature importance using Random Forest"""
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(self.X, self.y)
        self.feature_importance_rf = dict(zip(self.X.columns, rf.feature_importances_))
        return self.feature_importance_rf

    def compute_mutual_information(self) -> Dict[str, float]:
        """Compute mutual information scores between features and target"""
        self.mutual_info_scores = dict(
            zip(self.X.columns, mutual_info_classif(self.X, self.y))
        )
        return self.mutual_info_scores

    def analyze_temporal_patterns(self, time_window: int = 10) -> pd.DataFrame:
        """Analyze how feature values change over time windows"""
        temporal_data = []

        for feature in self.X.columns:
            feature_values = self.X[feature].values
            for window in range(0, len(feature_values) - time_window, time_window):
                window_data = feature_values[window : window + time_window]
                temporal_data.append(
                    {
                        "feature": feature,
                        "window": window // time_window,
                        "mean": np.mean(window_data),
                        "std": np.std(window_data),
                        "trend": np.polyfit(range(len(window_data)), window_data, 1)[0],
                    }
                )

        self.temporal_patterns = pd.DataFrame(temporal_data)
        return self.temporal_patterns

    def get_top_features(self, n: int = 10, method: str = "rf") -> List[str]:
        """Get top n important features based on specified method"""
        if method == "rf":
            if self.feature_importance_rf is None:
                self.compute_feature_importance()
            scores = self.feature_importance_rf
        else:  # mutual information
            if self.mutual_info_scores is None:
                self.compute_mutual_information()
            scores = self.mutual_info_scores

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n]

    def analyze_feature_correlations(self, features: List[str] = None) -> pd.DataFrame:
        """Analyze correlations between selected features"""
        if features is None:
            features = self.get_top_features(10)
        return self.X[features].corr()

    def generate_feature_evolution_plot(self, feature: str) -> go.Figure:
        """Generate plot showing how feature values evolve for different classes"""
        df_plot = pd.DataFrame(
            {"value": self.X[feature], "class": self.y, "index": range(len(self.y))}
        )

        fig = px.scatter(
            df_plot,
            x="index",
            y="value",
            color="class",
            title=f"Evolution of {feature} Values by Class",
            labels={"index": "Transaction Index", "value": "Feature Value"},
        )
        return fig

    def analyze_feature_distributions(
        self, features: List[str] = None
    ) -> Dict[str, go.Figure]:
        """Analyze value distributions for selected features by class"""
        if features is None:
            features = self.get_top_features(5)

        distributions = {}
        for feature in features:
            fig = px.histogram(
                pd.DataFrame({"value": self.X[feature], "class": self.y}),
                x="value",
                color="class",
                barmode="overlay",
                title=f"Distribution of {feature} by Class",
                labels={"value": "Feature Value", "count": "Frequency"},
            )
            distributions[feature] = fig

        return distributions

    def find_feature_interactions(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Find potentially important feature interactions"""
        top_features = self.get_top_features(top_n)
        interactions = []

        for i, f1 in enumerate(top_features):
            for f2 in top_features[i + 1 :]:
                # Create interaction feature
                interaction = self.X[f1] * self.X[f2]
                # Compute mutual information with target
                mi_score = mutual_info_classif(
                    interaction.values.reshape(-1, 1), self.y
                )[0]
                interactions.append((f1, f2, mi_score))

        return sorted(interactions, key=lambda x: x[2], reverse=True)
