import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Customer Purchase Predictor",
    page_icon="🛒",
    layout="centered"
)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #2ecc71;'>🛒 Customer Purchase Predictor</h1>
    <p style='text-align: center; color: gray;'>Will this customer make a purchase?</p>
    <hr>
""", unsafe_allow_html=True)

# Train model with sample data
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Age': np.random.randint(18, 70, n),
        'AnnualIncome': np.random.randint(20000, 150000, n),
        'NumberOfPurchases': np.random.randint(0, 50, n),
        'ProductCategory': np.random.randint(0, 5, n),
        'TimeSpentOnWebsite': np.random.uniform(1, 60, n),
        'LoyaltyProgram': np.random.randint(0, 2, n),
        'DiscountsAvailed': np.random.randint(0, 10, n),
    })
    df['PurchaseStatus'] = (
        (df['TimeSpentOnWebsite'] > 30) |
        (df['AnnualIncome'] > 80000) |
        (df['LoyaltyProgram'] == 1)
    ).astype(int)

    X = df.drop('PurchaseStatus', axis=1)
    y = df['PurchaseStatus']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Input form
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    annual_income = st.number_input("Annual Income ($)", 20000, 150000, 60000, step=1000)
    num_purchases = st.slider("Number of Past Purchases", 0, 50, 10)
    loyalty = st.selectbox("Loyalty Program Member", ["No", "Yes"])

with col2:
    product_category = st.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Home Goods", "Beauty", "Sports"]
    )
    time_spent = st.slider("Time Spent on Website (mins)", 1, 60, 20)
    discounts = st.slider("Discounts Availed", 0, 10, 2)

# Encode inputs
category_map = {"Electronics": 0, "Clothing": 1, "Home Goods": 2, "Beauty": 3, "Sports": 4}
loyalty_map = {"No": 0, "Yes": 1}

input_data = np.array([[
    age,
    annual_income,
    num_purchases,
    category_map[product_category],
    time_spent,
    loyalty_map[loyalty],
    discounts
]])

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Predict Purchase", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction == 1:
        st.success(f"✅ This customer is LIKELY to PURCHASE!")
        st.metric("Purchase Probability", f"{probability:.1f}%")
    else:
        st.error(f"❌ This customer is UNLIKELY to purchase.")
        st.metric("Purchase Probability", f"{probability:.1f}%")

    # Feature importance
    st.markdown("### 📊 Key Factors")
    features = ['Age', 'Annual Income', 'Past Purchases',
                'Product Category', 'Time on Website',
                'Loyalty Program', 'Discounts']
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    st.bar_chart(imp_df.set_index('Feature'))

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: gray;'>
    Built by Sana | B.Tech CSE, Indur Institute of Engineering & Technology
    </p>
""", unsafe_allow_html=True)
