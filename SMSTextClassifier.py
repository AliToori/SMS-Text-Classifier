import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import urllib.request
import os


class HealthCostPredictor:
    """A class to predict healthcare costs using TensorFlow Linear Regression."""

    def __init__(self, data_url):
        """Initialize with dataset URL and set up attributes."""
        self.data_url = data_url
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        self.model = None
        self.le_sex = LabelEncoder()
        self.le_smoker = LabelEncoder()
        self.le_region = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self):
        """Download and load the insurance dataset."""
        local_file = 'insurance.csv'
        try:
            if not os.path.exists(local_file):
                print("Downloading dataset...")
                urllib.request.urlretrieve(self.data_url, local_file)
            self.dataset = pd.read_csv(local_file)
            print("Dataset Head:")
            print(self.dataset.head())
            print("\nDataset Info:")
            print(self.dataset.info())
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def preprocess_data(self):
        """Encode categorical variables, split data, and normalize features."""
        if self.dataset is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        try:
            # Encode categorical columns
            self.dataset['sex'] = self.le_sex.fit_transform(self.dataset['sex'])
            self.dataset['smoker'] = self.le_smoker.fit_transform(self.dataset['smoker'])
            self.dataset['region'] = self.le_region.fit_transform(self.dataset['region'])

            # Separate features and target
            self.train_dataset = self.dataset.drop('expenses', axis=1)
            self.train_labels = self.dataset.pop('expenses')

            # Split data (80% train, 20% test)
            self.train_dataset, self.test_dataset, self.train_labels, self.test_labels = train_test_split(
                self.train_dataset, self.train_labels, test_size=0.2, random_state=42
            )

            # Normalize features
            self.train_dataset = self.scaler.fit_transform(self.train_dataset)
            self.test_dataset = self.scaler.transform(self.test_dataset)

            print("Train dataset shape:", self.train_dataset.shape)
            print("Test dataset shape:", self.test_dataset.shape)
            print("Train labels shape:", self.train_labels.shape)
            print("Test labels shape:", self.test_labels.shape)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess data: {e}")

    def build_model(self):
        """Build and compile the TensorFlow Linear Regression model."""
        try:
            self.model = tf.keras.Sequential([
                layers.Dense(units=1, input_shape=[self.train_dataset.shape[1]])
            ])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                loss='mean_absolute_error',
                metrics=['mae', 'mse']
            )
            print("Model built and compiled.")
        except Exception as e:
            raise RuntimeError(f"Failed to build model: {e}")

    def train_model(self):
        """Train the model on the training data."""
        if self.train_dataset is None or self.train_labels is None:
            raise ValueError("Training data not prepared. Run preprocess_data() first.")
        if self.model is None:
            raise ValueError("Model not built. Run build_model() first.")

        try:
            history = self.model.fit(
                self.train_dataset, self.train_labels,
                epochs=100,
                validation_split=0.2,
                verbose=0
            )
            print("Training completed. Final training MAE:", history.history['mae'][-1])
            return history
        except Exception as e:
            raise RuntimeError(f"Failed to train model: {e}")

    def evaluate_model(self):
        """Evaluate the model on test data and check MAE requirement."""
        if self.test_dataset is None or self.test_labels is None:
            raise ValueError("Test data not prepared. Run preprocess_data() first.")
        if self.model is None:
            raise ValueError("Model not built. Run build_model() first.")

        try:
            loss, mae, mse = self.model.evaluate(self.test_dataset, self.test_labels, verbose=2)
            print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

            if mae < 3500:
                print("You passed the challenge. Great job!")
            else:
                print("The Mean Abs Error must be less than 3500. Keep trying.")

            return mae
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate model: {e}")

    def visualize_results(self):
        """Visualize predicted vs actual expenses."""
        if self.test_dataset is None or self.test_labels is None:
            raise ValueError("Test data not prepared. Run preprocess_data() first.")
        if self.model is None:
            raise ValueError("Model not built. Run build_model() first.")

        try:
            test_predictions = self.model.predict(self.test_dataset).flatten()
            plt.figure(figsize=(8, 8))
            plt.scatter(self.test_labels, test_predictions, alpha=0.5)
            plt.xlabel('True values (expenses)')
            plt.ylabel('Predictions (expenses)')
            lims = [0, 50000]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims, 'r--')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Predicted vs Actual Healthcare Costs')
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Failed to visualize results: {e}")

    def run(self):
        """Run the full pipeline."""
        try:
            self.load_data()
            self.preprocess_data()
            self.build_model()
            self.train_model()
            self.evaluate_model()
            self.visualize_results()
        except Exception as e:
            print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    url = 'https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv'
    predictor = HealthCostPredictor(url)
    predictor.run()