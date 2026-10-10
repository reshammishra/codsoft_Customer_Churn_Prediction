# Customer Churn Prediction (CodSoft Task 3)

An end-to-end Machine Learning project to predict whether a customer will churn or stay with a bank using Scikit-Learn and Streamlit.

## 📌 Project Overview
- **Objective:** Predict bank customer churn (`Exited` = 1 vs 0) using demographic, financial, and account usage features.
- **Model:** Logistic Regression pipeline with `ColumnTransformer` (StandardScaler for numerical features, OneHotEncoder for categorical features).
- **Evaluation:** Evaluated using Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC.
- **Interactive App:** Streamlit web interface with real-time churn probability prediction.

## 📂 Project Structure
```text
├── Churn_Modelling.csv               # Dataset
├── customer_churn_prediction.ipynb   # Complete ML training & evaluation notebook
├── churn_model.pkl                   # Saved model pipeline
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/reshammishra/codsoft_Customer_Churn_Prediction.git
   cd codsoft_Customer_Churn_Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.
