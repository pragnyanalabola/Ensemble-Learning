
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sales Prediction - Ensemble Learning", layout="wide")
st.title("📈 Ensemble Learning: Predicting High Sales with Random Forest & XGBoost")

# Generate synthetic sales dataset
@st.cache_data
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

    # Generate target: High sales if total spend > threshold
    data['Total_Spend'] = data['Advertising'] + data['Promotion'] + data['Online_Spend'] + data['Retail_Spend']
    data['High_Sales'] = (data['Total_Spend'] > data['Total_Spend'].median()).astype(int)

    X = data.drop(['Total_Spend', 'High_Sales'], axis=1)
    y = data['High_Sales']
    return X, y

X, y = generate_sales_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

# Feature importance
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Streamlit layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Accuracy")
    st.write(f"✅ **Random Forest Accuracy:** `{rf_acc:.4f}`")
    st.write(f"✅ **XGBoost Accuracy:** `{xgb_acc:.4f}`")

with col2:
    st.subheader("📊 Top Feature Importance (Top 5)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    rf_importance.head(5).plot(kind='barh', ax=ax[0], color='skyblue')
    ax[0].set_title("Random Forest")
    ax[0].invert_yaxis()

    xgb_importance.head(5).plot(kind='barh', ax=ax[1], color='salmon')
    ax[1].set_title("XGBoost")
    ax[1].invert_yaxis()

    st.pyplot(fig)

st.markdown("---")
st.caption("🧠 Developed for ML Lab - Sales Classification with Ensemble Models")