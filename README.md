<<<<<<< HEAD
# 💬 Comment Toxicity Detection using Deep Learning

## 📌 Project Overview

This project aims to automatically detect toxic comments using a Deep Learning model built with TensorFlow/Keras. The model analyzes user comments and predicts whether they are **Toxic** or **Non-Toxic**.

The application is deployed using **Streamlit**, allowing users to:

* Predict toxicity for individual comments
* Upload CSV files for bulk prediction
* View model performance metrics
* Explore dataset insights

---

## 🎯 Problem Statement

Online platforms often struggle with toxic comments, abusive language, and hate speech. Manual moderation is time-consuming and inefficient.

This project uses Natural Language Processing (NLP) and Deep Learning to automatically classify comments and assist in content moderation.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-Learn
* Matplotlib
* Streamlit
* Joblib

---

## 📂 Project Structure

```text
Comment_Toxicity/
│
├── app.py
├── toxicity_model.h5
├── tokenizer.pkl
├── train.csv
├── requirements.txt
├── README.md
│
└── notebooks/
    └── Comment_Toxicity_Analysis.ipynb
```

---

## 📊 Dataset

The dataset contains user comments and toxicity labels.

### Features

| Column       | Description                               |
| ------------ | ----------------------------------------- |
| comment_text | User comment text                         |
| toxic        | Toxicity label (0 = Non-Toxic, 1 = Toxic) |

---

## ⚙️ Workflow

### 1. Data Collection

* Load dataset
* Explore data structure
* Check missing values

### 2. Data Preprocessing

* Convert text to lowercase
* Remove URLs
* Remove special characters
* Remove extra spaces

### 3. Feature Engineering

* Tokenization
* Text Sequencing
* Padding Sequences

### 4. Model Building

LSTM (Long Short-Term Memory) Neural Network

Architecture:

```text
Embedding Layer
        ↓
LSTM Layer
        ↓
Dense Layer
        ↓
Output Layer (Sigmoid)
```

### 5. Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

## 📈 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 95.39% |
| Precision | 76%    |
| Recall    | 76%    |

---

## 🚀 Streamlit Features

### 🔍 Single Comment Prediction

Enter a comment and instantly receive:

* Toxic / Non-Toxic classification
* Toxicity score
* Risk level

### 📂 Bulk Prediction

Upload a CSV file containing:

```text
comment_text
```

The application generates:

* Toxicity Score
* Prediction Label

and allows downloading the results.

### 📊 Dataset Insights

Displays:

* Total comments
* Toxic comments
* Non-toxic comments
* Distribution chart

### 🧪 Sample Test Cases

Built-in examples for quick testing.

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Comment_Toxicity.git
```

### Navigate to Project

```bash
cd Comment_Toxicity
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

Features include:

* Toxicity Detection Dashboard
* Dataset Insights
* Model Performance Metrics
* Bulk Prediction System

---

## 💡 Future Improvements

* Multi-label toxicity detection
* Hate speech classification
* Profanity filtering
* Word Cloud visualization
* Advanced NLP preprocessing
* Cloud deployment

---

## 👨‍💻 Author

**Sagar Bhawal**

Aspiring Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/Sagar-Bhawal

---

## 📜 License

This project is developed for educational and portfolio purposes.
=======

