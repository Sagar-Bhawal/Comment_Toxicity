import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PAGE CONFIG

st.set_page_config(
    page_title="Comment Toxicity Detector",
    page_icon="💬",
    layout="wide"
)

# SIDEBAR

st.sidebar.title("📌 About")

st.sidebar.info(
    """
    Comment Toxicity Detection System

    Model: LSTM
    Framework: TensorFlow
    Deployment: Streamlit

    Detects whether a comment is toxic or non-toxic.
    """
)

# LOAD MODEL & TOKENIZER

model = load_model("toxicity_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

MAX_LEN = 200

# TEXT CLEANING FUNCTION

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# PREDICTION FUNCTION

def predict_toxicity(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    probability = model.predict(padded, verbose=0)[0][0]
    return probability

# TITLE

st.title("💬 Comment Toxicity Detector")

st.write("""
    Enter a comment below and the model will predict
    whether it is toxic or non-toxic.
    """)

# SINGLE COMMENT PREDICTION

st.subheader("🔍 Single Comment Prediction")
comment = st.text_area("Enter Comment")

if st.button("Predict"):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        probability = predict_toxicity(comment)

        score = round(probability * 100, 2)
        if probability > 0.5:
            st.error("⚠️ Toxic Comment")
        else:
            st.success("✅ Non-Toxic Comment")
        st.metric("Toxicity Score", f"{score}%")

        if score <= 30:
            st.success("Safe")
        elif score <= 60:
            st.warning("Moderate")
        else:
            st.error("Highly Toxic")

# BULK CSV PREDICTION

st.subheader("📂 Bulk Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "comment_text" not in df.columns:

        st.error("CSV must contain a 'comment_text' column.")

    else:

        with st.spinner("Generating predictions..."):

            df["Toxicity_Score"] = df["comment_text"].apply(lambda x: round(predict_toxicity(str(x)) * 100, 2))

            df["Prediction"] = df["Toxicity_Score"].apply(lambda x: "Toxic"
                if x > 50
                else "Non-Toxic")

        st.success("Prediction Completed!")

        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(label="⬇ Download Results", data=csv, file_name="toxicity_predictions.csv", mime="text/csv")

# MODEL PERFORMANCE

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy","95.39%")

col2.metric("Precision","76%")

col3.metric("Recall","76%")

# DATASET INSIGHTS

try:

    dataset = pd.read_csv("train.csv")

    st.subheader("📈 Dataset Insights")

    total_comments = len(dataset)

    toxic_comments = int(dataset["toxic"].sum())

    non_toxic_comments = (total_comments - toxic_comments)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Comments", total_comments)

    c2.metric("Toxic Comments", toxic_comments)

    c3.metric("Non-Toxic Comments",non_toxic_comments)

    st.subheader("📊 Toxic vs Non-Toxic Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    dataset["toxic"].value_counts().plot(kind="bar", ax=ax)

    ax.set_title("Toxic vs Non-Toxic Comments")

    ax.set_xlabel("Class")

    ax.set_ylabel("Count")

    st.pyplot(fig)

except Exception:

    st.warning("train.csv not found. Dataset insights unavailable.")


# SAMPLE TEST CASES

st.subheader("🧪 Sample Test Cases")

sample_comments = [

    "Thank you for helping me",

    "You are an idiot",

    "Have a nice day",

    "Nobody likes you"

]

for sample in sample_comments:

    score = predict_toxicity(sample)

    label = ("Toxic"
        if score > 0.5
        else "Non-Toxic")

    st.write(f"**Comment:** {sample}")

    if label == "Toxic":

        st.error(f"Prediction: {label}")

    else:

        st.success(f"Prediction: {label}")

    st.write(f"Toxicity Score: {round(score * 100, 2)}%")

    st.write("---")


# FOOTER

st.markdown("---")

st.caption("Deep Learning for Comment Toxicity Detection using LSTM and Streamlit")

