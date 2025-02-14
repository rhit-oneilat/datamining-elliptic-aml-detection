import pandas as pd
import networkx as nx


class DataLoader:
    def __init__(self):
        self.edge_df = None
        self.class_df = None
        self.feature_df = None
        self.graph = None

    def load_data(self, edge_path, class_path, feature_path=None):
        """Load the dataset from CSV files"""
        self.edge_df = pd.read_csv(edge_path)
        self.class_df = pd.read_csv(class_path)

        if feature_path:
            # Generate column names for features dataset
            feature_columns = ["txId"] + [
                f"feature_{i}"
                for i in range(1, len(pd.read_csv(feature_path, nrows=1).columns))
            ]
            self.feature_df = pd.read_csv(feature_path, names=feature_columns)

            # Merge features with class data
            self.feature_df = self.feature_df.merge(
                self.class_df[["txId", "class"]], on="txId", how="left"
            )

        # Rename columns for consistency
        self.edge_df.rename(
            columns={"txId1": "source", "txId2": "target"}, inplace=True
        )

    def create_graph(self):
        """Create directed graph from edge data"""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.class_df["txId"])
        self.graph.add_edges_from(self.edge_df[["source", "target"]].values)
        return self.graph

    def merge_data(self):
        """Merge edge and class data"""
        merged_df = self.edge_df.merge(
            self.class_df, left_on="source", right_on="txId", how="left"
        )
        merged_df = merged_df.merge(
            self.class_df,
            left_on="target",
            right_on="txId",
            how="left",
            suffixes=("_source", "_target"),
        )
        return merged_df

    def get_labeled_data(self):
        """Get labeled data from feature dataset"""
        if self.feature_df is None:
            raise ValueError("Feature dataset not loaded")
        labeled_data = self.feature_df[self.feature_df["class"] != "unknown"].copy()
        return labeled_data

    def get_feature_matrix(self):
        """Get feature matrix X and labels y for model training"""
        if self.feature_df is None:
            raise ValueError("Feature dataset not loaded")

        labeled_data = self.get_labeled_data()

        # Select only feature columns and drop txId
        feature_cols = [
            col for col in labeled_data.columns if col.startswith("feature_")
        ]
        X = labeled_data[feature_cols]
        y = labeled_data["class"]

        return X, y
