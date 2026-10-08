
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="AI Customer Support Classifier",
    page_icon="🤖",
    layout="centered"
)

CONFIDENCE_THRESHOLD = 0.35

@st.cache_resource
def load_model():
    dataset_path = Path(__file__).parent / "customer_support_tickets.csv"

    data = pd.read_csv(dataset_path)

    required = {"customer_message", "category"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset columns are missing.")

    data = data.dropna(subset=["customer_message", "category"])

    if data.empty or data["category"].nunique() < 2:
        raise ValueError("Not enough valid training data.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(data["customer_message"], data["category"])
    return model

st.title("🤖 AI Customer Support Classifier")
st.write("SWYNEX Technologies Internship - Final AI Application")

st.markdown(
    "Enter a customer support message and let AI predict "
    "the correct support category."
)

try:
    model = load_model()
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

st.subheader("Classify a Customer Message")

message = st.text_area(
    "Customer Message",
    placeholder="Example: My package has not arrived yet",
    height=120
)

if st.button("Classify Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a valid customer message.")

    else:
        try:
            probabilities = model.predict_proba([message])[0]
            best_index = probabilities.argmax()

            category = model.classes_[best_index]
            confidence = float(probabilities[best_index])

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning("Needs Human Review")
                st.write(f"Suggested Category: {category}")
                st.info(
                    "The AI model is uncertain about this "
                    "message. A human support agent should review it."
                )
            else:
                st.success(f"Predicted Category: {category}")

        except (ValueError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.subheader("Supported Categories")
st.write(
    "Payment Issue | Delivery Issue | Product Issue | "
    "Refund / Return | Other"
)

st.caption(
    "Educational AI prototype using TF-IDF and Logistic Regression. "
    "Predictions should be reviewed before real-world use."
)
