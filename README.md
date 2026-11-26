# Flipkart Customer Service Satisfaction Prediction

This project predicts CSAT scores (1 to 5) for Flipkart customer support tickets using machine learning. It includes data cleaning, exploratory analysis, hypothesis testing, and two models (tuned Random Forest and Logistic Regression).

**Notebook:** `Flipkart_Capstone.ipynb` ([open on Google Colab](https://colab.research.google.com/drive/1t-ghiiySXMCafW6xNrcXSkKPlSwdv1i2))

**Dataset:** `Customer_support_data.csv`

**Live app:** https://flipkart-capstone-hnc7zz7akxsvsqd8ojqucs.streamlit.app

## How to run the app locally

1. Install the packages: `pip install streamlit scikit-learn joblib pandas numpy`
2. Run: `python -m streamlit run app.py`

The model downloads automatically on first run from Hugging Face.

## Results

- Best model: tuned Random Forest, macro F1 0.24, accuracy 52.5%
- Logistic Regression: macro F1 0.22, higher recall for unhappy customers
- Strongest single input: response time (correlation -0.16 with CSAT)
