# SMS Text Classifier using TensorFlow Sequential Neural Network

This project implements an SMS text classification system using a Sequential Neural Network with Embedding and Dense Layers in TensorFlow, developed as part of the freeCodeCamp Machine Learning with Python certification. The goal is to classify SMS messages as either "ham" (normal messages) or "spam" (advertisements or company messages) using the SMS Spam Collection dataset.

---

👨‍💻 **Author**: Ali Toori – Full-Stack Python Developer  
📺 **YouTube**: [@AliToori](https://youtube.com/@AliToori)  
💬 **Telegram**: [@AliToori](https://t.me/@AliToori)  
📂 **GitHub**: [Github.com/AliToori](https://github.com/AliToori)

---

### Project Overview
The project involves:
1. Loading and preprocessing the SMS Spam Collection dataset, converting text messages into padded sequences and labels into binary (0 for ham, 1 for spam).
2. Building a **Sequential Neural Network** using TensorFlow’s `Sequential` API, featuring:
   - An **Embedding Layer** (16 dimensions) to convert tokenized words into dense vectors.
   - A **GlobalAveragePooling1D** layer to aggregate embeddings.
   - A **Dense Layer** with 24 units and ReLU activation for pattern learning.
   - A **Dense Output Layer** with 1 unit and Sigmoid activation for binary classification.
3. Training the model with binary cross-entropy loss and Adam optimizer to classify messages as "ham" or "spam" with high accuracy.
4. Implementing a `predict_message` function that takes a message string and returns a list containing the probability of "spam" (0 to 1) and the predicted label ("ham" or "spam").
5. Testing the model against a set of predefined messages to ensure correct classification.
6. Providing a modular, class-based Python script (`SMSTextClassifier.py`) for reusability.

Example output from the prediction function:
```python
[0.008318834938108921, 'ham']
```

Example test result:
```bash
You passed the challenge. Great job!
```

---

### [Google Colab Project Link](https://colab.research.google.com/drive/1XF7scLFr_brcQARJEf5CiS3lxFcaSSrr)

---

## 🛠 Tech Stack
* Language: Python 3.10+
* Libraries:
  * TensorFlow (for building a Sequential Neural Network with Embedding, GlobalAveragePooling1D, and Dense Layers)
  * Pandas (for data loading and preprocessing)
  * NumPy (for numerical operations)
  * Requests (for downloading datasets in Python script)
* Tools:
  * Google Colab for development, training, and testing (with GPU support)
  * GitHub for version control

---

## 📂 Project Structure
The project includes:
* `SMSTextClassifier.py`: A class-based Python script with methods for downloading, preprocessing, building, training, and testing the Sequential Neural Network model.
* `AliTori_sms_text_classification.ipynb`: Colab notebook implementing the logic in five cells (import libraries, load data, preprocess/train, define prediction function, test predictions).
* `train-data.tsv`: Training dataset (downloaded from [https://cdn.freecodecamp.org/project-data/sms/train-data.tsv](https://cdn.freecodecamp.org/project-data/sms/train-data.tsv)).
* `valid-data.tsv`: Test dataset (downloaded from [https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv](https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv)).
* `README.md`: This file.

Dataset structure:
```bash
train-data.tsv: Contains columns [label (ham/spam), message]
valid-data.tsv: Contains columns [label (ham/spam), message]
```

---

## Usage
### Python Script
1. Save `SMSTextClassifier.py` locally.
2. Install required libraries:
   ```bash
   pip install tensorflow pandas numpy requests
   ```
3. Run the script:
   ```bash
   python SMSTextClassifier.py
   ```
4. The script will:
   - Download the datasets if not present
   - Preprocess data, train the Sequential Neural Network, and test predictions
   - Output the prediction for a sample message and the test results

### Colab Notebook
1. Open the Colab notebook: [Google Colab Project Link](https://colab.research.google.com/drive/1XF7scLFr_brcQARJEf5CiS3lxFcaSSrr)
2. Save a copy to your Google Drive (**File > Save a copy in Drive**).
3. Enable GPU for faster training (**Runtime > Change runtime type > GPU**).
4. Run all cells sequentially:
   - Cell 1: Import libraries and install `tensorflow`
   - Cell 2: Download and load the datasets
   - Cell 3: Preprocess data, build and train the Sequential Neural Network
   - Cell 4: Define `predict_message` function
   - Cell 5: Test the model with predefined messages
5. Ensure the notebook’s sharing settings are “anyone with the link” for submission.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository: [https://github.com/AliToori/SMS-Text-Classifier](https://github.com/AliToori/SMS-Text-Classifier)
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.
Alternatively, share an updated Colab notebook link via GitHub issues or Telegram.

---

## 🙏 Acknowledgments
- Built as part of the [freeCodeCamp Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python) certification.
- Uses TensorFlow for developing a Sequential Neural Network with Embedding and Dense Layers, and Google Colab for cloud-based execution.
- Special thanks to freeCodeCamp for providing the challenge framework and dataset.

## 🆘 Support
For questions, issues, or feedback:  
📺 YouTube: [@AliToori](https://youtube.com/@AliToori)  
💬 Telegram: [@AliToori](https://t.me/@AliToori)  
📂 GitHub: [github.com/AliToori](https://github.com/AliToori)