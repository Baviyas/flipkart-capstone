import json
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_URL = "https://drive.google.com/uc?export=download&id=1iuDt0esgOHuDaQGKrmzEeYkB6nKbbvV"
MODEL_FILE = "best_rf_model.joblib"

if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model (first run only)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

model = joblib.load(MODEL_FILE)
columns = json.load(open("feature_columns.json"))

st.title("Flipkart CSAT Predictor")
st.write("Enter a support ticket's details to predict the customer's satisfaction score (1 to 5).")

channel = st.selectbox("Channel", ["Inbound", "Outcall", "Email"])
category = st.selectbox("Category", ["Returns", "Cancellation", "Feedback", "Order Related",
                                     "Payments related", "Product Queries", "Refund Related",
                                     "Shopzilla Related", "Offers & Cashback",
                                     "Onboarding related", "Others"])
shift = st.selectbox("Agent shift", ["Morning", "Evening", "Night", "Split", "Afternoon"])
response_hours = st.slider("Response time (hours)", 0.0, 24.0, 0.1, 0.05)

# Start with every input at 0, then switch on the ones the user chose
row = pd.DataFrame([[0.0] * len(columns)], columns=columns)
for name in [f"channel_name_{channel}", f"category_{category}", f"Agent Shift_{shift}"]:
    if name in row.columns:
        row[name] = 1.0
row["response_time_log"] = np.log1p(response_hours) 

probs = model.predict_proba(row)[0]
labels = model.classes_
best = labels[probs.argmax()]

st.subheader(f"Predicted CSAT score: {best}")
st.bar_chart(pd.Series(probs, index=[str(c) for c in labels]))
