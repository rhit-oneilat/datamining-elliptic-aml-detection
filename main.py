"""
Streamlit frontend for Elliptic Transaction Classification
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import io
import plotly.express as px
from data.data_loader import DataLoader
from features.feature_analysis import FeatureAnalyzer
from features.graph_features import GraphFeatureExtractor
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from config.model_config import MAX_HOPS, TEST_SIZE, RANDOM_STATE

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_exploration_section(X, y):
    """Create the feature exploration section of the interface"""
    st.header("Feature Exploration")

    # Initialize feature analyzer
    analyzer = FeatureAnalyzer(X, y)

    # Sidebar controls for exploration
    exploration_type = st.sidebar.selectbox(
        "Exploration Type",
        [
            "Feature Importance",
            "Temporal Analysis",
            "Feature Distributions",
            "Feature Correlations",
            "Feature Interactions",
        ],
    )

    if exploration_type == "Feature Importance":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Random Forest Importance")
            rf_importance = analyzer.compute_feature_importance()
            top_rf = analyzer.get_top_features(10, "rf")

            # Plot RF importance
            fig_rf = px.bar(
                x=[rf_importance[f] for f in top_rf],
                y=top_rf,
                orientation="h",
                title="Top 10 Features (Random Forest)",
                labels={"x": "Importance", "y": "Feature"},
            )
            st.plotly_chart(fig_rf)

        with col2:
            st.subheader("Mutual Information Scores")
            mi_scores = analyzer.compute_mutual_information()
            top_mi = analyzer.get_top_features(10, "mi")

            # Plot MI scores
            fig_mi = px.bar(
                x=[mi_scores[f] for f in top_mi],
                y=top_mi,
                orientation="h",
                title="Top 10 Features (Mutual Information)",
                labels={"x": "Mutual Information", "y": "Feature"},
            )
            st.plotly_chart(fig_mi)

    elif exploration_type == "Temporal Analysis":
        st.subheader("Temporal Patterns")

        # Time window selection
        window_size = st.slider("Time Window Size", 5, 50, 10)
        temporal_data = analyzer.analyze_temporal_patterns(window_size)

        # Feature selection for temporal analysis
        selected_feature = st.selectbox("Select Feature", analyzer.get_top_features(20))

        # Plot feature evolution
        fig_evolution = analyzer.generate_feature_evolution_plot(selected_feature)
        st.plotly_chart(fig_evolution)

    elif exploration_type == "Feature Distributions":
        st.subheader("Feature Distributions")

        # Feature selection for distribution analysis
        n_features = st.slider("Number of Features", 1, 10, 5)
        distributions = analyzer.analyze_feature_distributions(
            analyzer.get_top_features(n_features)
        )

        # Display distributions
        for feature, fig in distributions.items():
            st.plotly_chart(fig)

    elif exploration_type == "Feature Correlations":
        st.subheader("Feature Correlations")

        # Number of features for correlation analysis
        n_features = st.slider("Number of Features", 5, 20, 10)
        corr_matrix = analyzer.analyze_feature_correlations(
            analyzer.get_top_features(n_features)
        )

        # Plot correlation matrix
        fig_corr = px.imshow(
            corr_matrix, title="Feature Correlations", color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr)

    elif exploration_type == "Feature Interactions":
        st.subheader("Feature Interactions")

        # Number of top features to consider
        n_features = st.slider("Number of Top Features", 3, 10, 5)
        interactions = analyzer.find_feature_interactions(n_features)

        # Display interactions
        st.write("Top Feature Interactions:")
        for f1, f2, score in interactions:
            st.write(f"{f1} × {f2}: {score:.4f}")


def load_data(data_dir="data"):
    """Load data with error handling and caching"""

    @st.cache_data
    def _load_data(data_dir):
        try:
            data_paths = {
                "edge_path": Path(data_dir) / "elliptic_txs_edgelist.csv",
                "class_path": Path(data_dir) / "elliptic_txs_classes.csv",
                "feature_path": Path(data_dir) / "elliptic_txs_features.csv",
            }

            data_loader = DataLoader()
            data_loader.load_data(**data_paths)
            return data_loader
        except FileNotFoundError as e:
            st.error(f"Error loading data: {e}")
            return None

    return _load_data(data_dir)


def process_graph_features(data_loader, feature_extractor):
    """Process and extract graph-based features"""
    with st.spinner("Processing graph-based features..."):
        merged_df = data_loader.merge_data()

        # Extract features for both classes
        class_1_features = feature_extractor.extract_features_for_class(
            merged_df, node_class=1
        )
        class_2_features = feature_extractor.extract_features_for_class(
            merged_df, node_class=2
        )

        # Combine features into a DataFrame
        all_features = pd.DataFrame.from_dict(
            {**class_1_features, **class_2_features}, orient="index"
        )
        all_features["class"] = ["1"] * len(class_1_features) + ["2"] * len(
            class_2_features
        )

        # Prepare X and y
        X = all_features.drop(columns=["class"])
        y = all_features["class"].astype(int)

        return X, y


def process_opaque_features(data_loader):
    """Process the original opaque features"""
    with st.spinner("Processing opaque features..."):
        labeled_data = data_loader.get_labeled_data()

        # Prepare features (excluding txId and class)
        X = labeled_data.drop(["class"])  # 1 is the txId column
        y = labeled_data["class"]

        # Convert features to float
        cols_to_convert = [i for i in range(2, 168)]
        X[cols_to_convert] = X[cols_to_convert].astype(float)

        return X, y


def plot_confusion_matrix(cm, classes):
    """Create a plotly confusion matrix figure"""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis={"side": "bottom"},
    )
    return fig


def create_feature_importance_plot(model, feature_names, n_top_features=10):
    """Create feature importance plot"""
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            indices = importances.argsort()[-n_top_features:][::-1]

            fig = go.Figure(
                go.Bar(
                    x=importances[indices], y=feature_names[indices], orientation="h"
                )
            )

            fig.update_layout(
                title=f"Top {n_top_features} Most Important Features",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=400,
            )

            return fig
    return None


def main():
    st.set_page_config(page_title="Elliptic Transaction Classification", layout="wide")

    st.title("Elliptic Transaction Classification")

    # Sidebar
    st.sidebar.header("Configuration")
    feature_type = st.sidebar.selectbox(
        "Feature Type",
        ["graph", "opaque"],
        help="Choose the type of features to use for classification",
    )

    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Random Forest", "Logistic Regression"],
        help="Choose the classification model to use",
    )

    # Main content
    data_loader = load_data()
    if data_loader is None:
        st.error("Failed to load data. Please check your data directory.")
        return

    # Process features based on selection
    try:
        if feature_type == "graph":
            graph = data_loader.create_graph()
            feature_extractor = GraphFeatureExtractor(graph, max_hops=MAX_HOPS)
            X, y = process_graph_features(data_loader, feature_extractor)
        else:
            X, y = process_opaque_features(data_loader)

        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", X.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Classes", len(set(y)))

        # Train model
        with st.spinner("Training model..."):
            model_class = (
                RandomForestModel
                if model_type == "Random Forest"
                else LogisticRegressionModel
            )
            model = model_class(random_state=RANDOM_STATE)
            model.prepare_data(X, y, test_size=TEST_SIZE)
            training_results = model.train()
            evaluation_results = model.evaluate()

        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.text("Classification Report")
            st.code(evaluation_results["classification_report"])

        with col2:
            st.text("Confusion Matrix")
            cm = evaluation_results["confusion_matrix"]
            fig_cm = plot_confusion_matrix(cm, classes=["Class 1", "Class 2"])
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importance plot (for Random Forest)
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            fig_importance = create_feature_importance_plot(
                model.model, pd.Index(X.columns), n_top_features=10
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        # Add exploration section after model training
        if st.checkbox("Show Feature Exploration"):
            create_exploration_section(X, y)
        # Download results
        if st.button("Download Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{feature_type}_{model_type}_{timestamp}_results.txt"

            buffer = io.StringIO()
            buffer.write(f"Results for {model_type} using {feature_type} features\n")
            buffer.write("=" * 50 + "\n\n")

            buffer.write("Training Results:\n")
            buffer.write("-" * 20 + "\n")
            for key, value in training_results.items():
                buffer.write(f"{key}: {value}\n")

            buffer.write("\nEvaluation Results:\n")
            buffer.write("-" * 20 + "\n")
            buffer.write("Classification Report:\n")
            buffer.write(evaluation_results["classification_report"])
            buffer.write("\nConfusion Matrix:\n")
            buffer.write(str(evaluation_results["confusion_matrix"]))

            st.download_button(
                label="Download Results as Text",
                data=buffer.getvalue(),
                file_name=filename,
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Error in main execution")


if __name__ == "__main__":
    main()
