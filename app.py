"""
H&M Sales Dashboard - Simplified Version
Features: Data Overview, EDA, Model Metrics, Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="H&M Sales Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio("Select Page:", 
                    ["🏠 Overview",
                     "📊 Data Analysis", 
                     "🤖 Model Performance",
                     "🔮 Predictions"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Generate sample data
@st.cache_data
def load_data(n=1000):
    np.random.seed(42)
    return pd.DataFrame({
        'Product_ID': [f'P{i:05d}' for i in range(1, n+1)],
        'Price': np.random.uniform(10, 200, n),
        'Product_Type': np.random.choice(['Dress', 'Shirt', 'Pants', 'Jacket', 'Shoes'], n),
        'Season': np.random.choice(['Summer', 'Winter', 'Spring', 'Fall'], n),
        'Brand_Score': np.random.uniform(5, 10, n),
        'Discount': np.random.uniform(0, 0.5, n),
        'Review_Score': np.random.uniform(1, 5, n),
        'Sales': np.random.randint(50, 1000, n),
        'Success': np.random.choice([0, 1], n, p=[0.3, 0.7])
    })

df = load_data()

# PAGE 1: OVERVIEW
if page == "🏠 Overview":
    st.title("🛍️ H&M Sales Analytics Dashboard")
    st.markdown("### ML Pipeline & Business Intelligence Platform")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", f"{len(df):,}")
    col2.metric("Total Sales", f"{df['Sales'].sum():,}")
    col3.metric("Success Rate", f"{(df['Success'].mean()*100):.1f}%")
    col4.metric("Avg Price", f"${df['Price'].mean():.2f}")
    
    st.markdown("---")
    
    # Project overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Project Pipeline")
        st.markdown("""
        **1. Data Cleaning**
        - Preprocessing & quality checks
        - Missing value handling
        - Feature engineering
        
        **2. Exploratory Analysis**
        - Statistical analysis
        - Visualization
        - Correlation studies
        
        **3. ML Modeling**
        - Model training & tuning
        - Performance evaluation
        - Model selection
        """)
    
    with col2:
        st.markdown("### 🔬 Key Features")
        st.markdown("""
        **4. Explainability**
        - SHAP feature importance
        - LIME local explanations
        - Model interpretability
        
        **5. Fairness Analysis**
        - Bias detection
        - Fairness metrics
        - Mitigation strategies
        
        **6. Deployment**
        - FastAPI endpoints
        - Docker containerization
        - Real-time predictions
        """)

# PAGE 2: DATA ANALYSIS
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis & Visualization")
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        price_range = st.slider("Price Range", 
                                float(df['Price'].min()), 
                                float(df['Price'].max()),
                                (float(df['Price'].min()), float(df['Price'].max())))
    with col2:
        product_types = st.multiselect("Product Type", 
                                       df['Product_Type'].unique(),
                                       default=df['Product_Type'].unique())
    with col3:
        seasons = st.multiselect("Season", 
                                df['Season'].unique(),
                                default=df['Season'].unique())
    
    # Apply filters
    df_filtered = df[
        (df['Price'] >= price_range[0]) &
        (df['Price'] <= price_range[1]) &
        (df['Product_Type'].isin(product_types)) &
        (df['Season'].isin(seasons))
    ]
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sales by Product Type")
        sales_by_type = df_filtered.groupby('Product_Type')['Sales'].sum()
        fig = px.bar(x=sales_by_type.index, y=sales_by_type.values,
                    color=sales_by_type.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Price Distribution")
        fig = px.histogram(df_filtered, x='Price', nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Success Rate by Season")
        success_by_season = df_filtered.groupby('Season')['Success'].mean() * 100
        fig = px.bar(x=success_by_season.index, y=success_by_season.values,
                    color=success_by_season.values, color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Price vs Sales")
        fig = px.scatter(df_filtered, x='Price', y='Sales', 
                        color='Product_Type', size='Review_Score')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### Filtered Data")
    st.dataframe(df_filtered.head(50), use_container_width=True)

# PAGE 3: MODEL PERFORMANCE
elif page == "🤖 Model Performance":
    st.title("🤖 ML Model Performance")
    st.markdown("---")
    
    # Model comparison data
    models_data = {
        'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'KNN', 'SVM'],
        'Accuracy': [100.0, 100.0, 100.0, 95.30, 94.73],
        'Precision': [100.0, 100.0, 100.0, 96.24, 94.24],
        'Recall': [100.0, 100.0, 100.0, 94.31, 95.25],
        'F1_Score': [100.0, 100.0, 100.0, 95.27, 94.74]
    }
    df_models = pd.DataFrame(models_data)
    
    st.markdown("### Model Comparison")
    
    # Metrics visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=df_models['Model'], y=df_models['Accuracy']))
    fig.add_trace(go.Bar(name='F1-Score', x=df_models['Model'], y=df_models['F1_Score']))
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.markdown("### Detailed Metrics")
    st.dataframe(df_models, use_container_width=True)
    
    st.markdown("---")
    
    # SHAP Feature Importance
    st.markdown("### Feature Importance (SHAP)")
    
    shap_data = {
        'Feature': ['Price', 'Product_Type', 'Brand_Score', 'Season', 'Discount', 'Review_Score'],
        'Importance': [0.24, 0.18, 0.15, 0.12, 0.09, 0.08]
    }
    df_shap = pd.DataFrame(shap_data)
    
    fig = px.bar(df_shap, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**Key Insight:** Price is the most influential feature (24% importance)")

# PAGE 4: PREDICTIONS
elif page == "🔮 Predictions":
    st.title("🔮 Product Success Predictor")
    st.markdown("---")
    
    st.info("Enter product details to predict success probability")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price = st.number_input("Price ($)", min_value=0.0, value=49.99)
            product_type = st.selectbox("Product Type", 
                ["Dress", "Shirt", "Pants", "Jacket", "Shoes"])
            season = st.selectbox("Season", 
                ["Summer", "Winter", "Spring", "Fall"])
        
        with col2:
            brand_score = st.slider("Brand Score", 0.0, 10.0, 7.5)
            discount = st.slider("Discount", 0.0, 0.5, 0.15)
            review_score = st.slider("Review Score", 1.0, 5.0, 4.0)
        
        with col3:
            st.markdown("###")
            st.markdown("###")
            predict_btn = st.form_submit_button("🎯 Predict", use_container_width=True)
        
        if predict_btn:
            # Simple prediction logic
            score = 0.5
            if price < 50:
                score += 0.15
            if brand_score > 7:
                score += 0.1
            if discount > 0.1:
                score += 0.08
            if review_score > 4:
                score += 0.1
            if season in ["Summer", "Spring"]:
                score += 0.05
            
            score = np.clip(score, 0, 1)
            prediction = 1 if score > 0.5 else 0
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("### ✅ SUCCESS")
                else:
                    st.error("### ❌ FAILURE")
            
            with col2:
                st.metric("Probability", f"{score:.1%}")
            
            with col3:
                expected_sales = int(score * 500)
                st.metric("Expected Sales", f"{expected_sales:,}")
            
            # Feature contributions
            st.markdown("### 📈 Key Contributing Factors")
            
            factors = {
                'Factor': ['Price', 'Brand Score', 'Discount', 'Review Score', 'Season'],
                'Impact': [
                    0.85 if price < 50 else 0.35,
                    brand_score / 10,
                    discount * 2,
                    review_score / 5,
                    0.7 if season in ["Summer", "Spring"] else 0.5
                ]
            }
            df_factors = pd.DataFrame(factors)
            
            fig = px.bar(df_factors, x='Impact', y='Factor', orientation='h',
                        color='Impact', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>H&M Sales Analytics Dashboard</strong> | Powered by Streamlit & ML</p>
    <p style='font-size: 12px;'>© 2024 H&M Analytics Team</p>
</div>
""", unsafe_allow_html=True)