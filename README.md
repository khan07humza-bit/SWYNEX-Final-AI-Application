
# SWYNEX Final AI Application – Task 4

## AI Customer Support Ticket Classifier

A beginner-friendly AI web application developed as Task 4 of the SWYNEX Technologies internship.

## Problem Statement

Customer support teams receive messages about payments, deliveries, products, refunds, and account-related issues. Manually sorting these messages takes time.

This prototype uses machine learning to suggest a support category.

## Target Users

Customer support teams and e-commerce businesses.

## Method

- Python and Streamlit for the web application
- Pandas for loading the dataset
- TF-IDF for text feature extraction
- Logistic Regression for classification
- A 35% confidence threshold for human review

## Dataset

50 synthetic labeled customer support messages, with 10 examples in each of five categories:

- Payment Issue
- Delivery Issue
- Product Issue
- Refund / Return
- Other

## Intelligent Feature

The model predicts a category and displays its confidence.

Predictions below 35% are marked **Needs Human Review**.

This threshold is illustrative, not a calibrated guarantee of correctness.

## Live Demo

https://swynex-ai-support-classifier.streamlit.app/

## How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`
3. Open the browser URL shown in the terminal.

## Example Inputs

- My package has not arrived yet
- My payment was deducted twice
- Please help me

## Evaluation Approach

Test messages from each category, compare predictions with expected labels, inspect confidence scores, and test ambiguous and empty inputs.

The app was manually tested; no independent held-out accuracy has been established. A larger dataset and separate test split are needed for reliable evaluation.

## Error Handling

The app warns users about empty input and handles missing or invalid dataset files and selected prediction errors.

## Limitations

- Only 50 synthetic training examples
- Only five categories
- May misclassify unfamiliar or ambiguous messages
- Confidence scores are not calibrated probabilities of correctness
- Not ready for production use

## Responsible AI and Ethics

- Do not submit real customer personal or payment details to the public demo.
- Human agents should review uncertain or sensitive tickets.
- Evaluate performance across different writing styles and languages before deployment.
- Avoid using predictions as the sole basis for denying refunds or support.
- Obtain permission and apply appropriate privacy safeguards before using real customer data.

## Project Files

- `app.py` – Streamlit web application and ML classifier
- `customer_support_tickets.csv` – Synthetic training dataset
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

## Internship

SWYNEX Technologies | Task 4 – Final AI Application

This project builds on Tasks 1, 2, and 3.
  
