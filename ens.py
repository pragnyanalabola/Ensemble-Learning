import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit page config
st.set_page_config(page_title="XGBoost Sales Prediction", layout="centered")
st.title("📈 XGBoost: Predicting High Sales")

# Generate synthetic sales dataset
def generate_sales_data(n=500):
    np.random.seed(42)
    data = pd.DataFrame({
        'Advertising': np.random.normal(20000, 5000, n),
        'Promotion': np.random.normal(10000, 3000, n),
        'Discount': np.random.normal(10, 3, n),
        'Online_Spend': np.random.normal(15000, 4000, n),
        'Retail_Spend': np.random.normal(12000, 3500, n),
        'Month': np.random.randint(1, 13, n),
    })

    data['Total_Spend'] = data['Advertising'] + data['Promotion'] + data['Online_Spend'] + data['Retail_Spend']
    data['High_Sales'] = (data['Total_Spend'] > data['Total_Spend'].median()).astype(int)

    X = data.drop(['Total_Spend', 'High_Sales'], axis=1)
    y = data['High_Sales']
    return X, y

# Load data
X, y = generate_sales_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions and accuracy
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)


xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display accuracy
st.write(f"✅ **XGBoost Accuracy:** `{xgb_acc:.4f}`")
    # Display feature importance chart
st.subheader("📊 Top 5 Feature Importances")
fig, ax = plt.subplots(figsize=(6, 5))

xgb_importance.head(5).plot(kind='barh', ax=ax, color='salmon')
ax.set_title("Top Features - XGBoost")
ax.invert_yaxis()
st.pyplot(fig)
# Footer
st.markdown("---")
st.caption("🧠 XGBoost Sales Classification")
