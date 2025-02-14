"""Base model class with common functionality"""

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class BaseModel(ABC):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, X, y, test_size=0.3):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

    @abstractmethod
    def train(self):
        """Train the model"""
        pass

    def evaluate(self):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.model.predict(self.X_test)

        results = {
            "classification_report": classification_report(self.y_test, y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
        }

        return results

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
