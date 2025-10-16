import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os


class SMSTextClassifier:
    """SMS Text Classifier using a TensorFlow Sequential Neural Network.

    Classifies SMS messages as 'ham' or 'spam' using the SMS Spam Collection dataset.
    Implements a Sequential Neural Network with Embedding, GlobalAveragePooling1D, and Dense Layers.
    """

    def __init__(self, vocab_size=10000, max_length=120, embedding_dim=16):
        """Initialize the classifier with model parameters.

        Args:
            vocab_size (int): Maximum number of words in the vocabulary. Defaults to 10000.
            max_length (int): Maximum length of input sequences. Defaults to 120.
            embedding_dim (int): Dimension of the embedding layer. Defaults to 16.
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.trunc_type = 'post'
        self.padding_type = 'post'
        self.oov_tok = "<OOV>"
        self.tokenizer = None
        self.model = None
        self.train_file_path = "train-data.tsv"
        self.test_file_path = "valid-data.tsv"

    def download_data(self):
        """Download training and test datasets from freeCodeCamp."""
        for file_path, url in [
            (self.train_file_path, "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"),
            (self.test_file_path, "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv")
        ]:
            if not os.path.exists(file_path):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {file_path}")
                except requests.RequestException as e:
                    raise RuntimeError(f"Failed to download {file_path}: {e}")

    def load_and_preprocess_data(self):
        """Load and preprocess SMS datasets.

        Loads train and test datasets, maps labels (ham: 0, spam: 1), tokenizes text, and pads sequences.
        """
        if not (os.path.exists(self.train_file_path) and os.path.exists(self.test_file_path)):
            raise FileNotFoundError("Dataset files not found. Run download_data() first.")

        # Load datasets
        train_data = pd.read_csv(self.train_file_path, sep='\t', header=None, names=['label', 'message'])
        test_data = pd.read_csv(self.test_file_path, sep='\t', header=None, names=['label', 'message'])

        # Map labels
        train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
        test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

        # Tokenize and pad sequences
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)
        self.tokenizer.fit_on_texts(train_data['message'])

        train_sequences = self.tokenizer.texts_to_sequences(train_data['message'])
        self.train_padded = pad_sequences(
            train_sequences, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type
        )
        test_sequences = self.tokenizer.texts_to_sequences(test_data['message'])
        self.test_padded = pad_sequences(
            test_sequences, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type
        )

        self.train_labels = np.array(train_data['label'])
        self.test_labels = np.array(test_data['label'])
        print("Data preprocessing completed")

    def build_model(self):
        """Build and compile the Sequential Neural Network.

        Layers:
        - Embedding: Converts words to dense vectors.
        - GlobalAveragePooling1D: Aggregates embeddings.
        - Dense (24 units, ReLU): Learns patterns.
        - Dense (1 unit, Sigmoid): Outputs probability for binary classification.
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Model built and compiled")

    def train_model(self, epochs=30):
        """Train the model on preprocessed data.

        Args:
            epochs (int): Number of training epochs. Defaults to 30.

        Raises:
            ValueError: If model or tokenizer is not initialized.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized. Run load_and_preprocess_data and build_model first.")

        history = self.model.fit(
            self.train_padded, self.train_labels,
            epochs=epochs,
            validation_data=(self.test_padded, self.test_labels),
            verbose=2
        )
        print("Model training completed")
        return history

    def predict_message(self, pred_text):
        """Predict if a message is 'ham' or 'spam'.

        Args:
            pred_text (str): The input message to classify.

        Returns:
            list: [probability of spam (float), label ('ham' or 'spam')]

        Raises:
            ValueError: If tokenizer or model is not initialized.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Tokenizer or model not initialized")

        sequence = self.tokenizer.texts_to_sequences([pred_text])
        padded = pad_sequences(
            sequence, maxlen=self.max_length, padding=self.padding_type, truncating=self.trunc_type
        )
        prediction = self.model.predict(padded, verbose=0)[0][0]
        label = 'spam' if prediction >= 0.5 else 'ham'
        return [float(prediction), label]

    def test_predictions(self):
        """Test the model with predefined messages.

        Returns:
            bool: True if all test cases pass, False otherwise.
        """
        test_messages = [
            "how are you doing today",
            "sale today! to stop texts call 98912460324",
            "i dont want to go. can we try it a different day? available sat",
            "our new mobile video service is live. just install on your phone to start watching.",
            "you have won £1000 cash! call to claim your prize.",
            "i'll bring it tomorrow. don't forget the milk.",
            "wow, is your arm alright. that happened to me one time too"
        ]
        test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
        passed = True

        for msg, ans in zip(test_messages, test_answers):
            prediction = self.predict_message(msg)
            if prediction[1] != ans:
                passed = False
                print(f"Test failed for message '{msg[:30]}...': Expected '{ans}', Got '{prediction[1]}'")

        print("You passed the challenge. Great job!" if passed else "You haven't passed yet. Keep trying.")
        return passed


if __name__ == "__main__":
    classifier = SMSTextClassifier()
    classifier.download_data()
    classifier.load_and_preprocess_data()
    classifier.build_model()
    classifier.train_model()
    print(classifier.predict_message("how are you doing today"))
    classifier.test_predictions()