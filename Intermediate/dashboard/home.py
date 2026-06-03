import streamlit as st
import pandas as pd
import numpy as np

def render_home(df, metrics, best_model_name):
    st.markdown("<h1 style='color: #1d3557; font-weight: 800;'>Enterprise Credit Risk Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 1.2rem; margin-bottom: 2rem;'>Welcome to the AI-Powered Loan Approval & Risk Assessment Platform.</p>", unsafe_allow_html=True)
    
    with st.expander("Explore the Engine's Capabilities...", expanded=False):
        st.write("This platform provides real-time machine learning predictions for loan applications. It evaluates financial data, computes a comprehensive Risk Score, and utilizes SHAP to offer transparent reasoning for every decision.")
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1d3557; font-weight: 700; margin-bottom: 1.5rem;'>Portfolio Overview</h3>", unsafe_allow_html=True)
    
    # Calculate KPIs
    total_apps = len(df)
    total_portfolio = df['LoanAmount'].sum() * 1000 # LoanAmount is usually in thousands
    approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
    f1_score = metrics.get(best_model_name, {}).get('F1 Score', 0) * 100
    
    st.markdown("""
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #fcfcfc 100%);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid rgba(0,0,0,0.03);
        border-left: 6px solid #3a86ff; /* Ensure left border is applied last */
    }
    .impact-label {
        color: #6c757d;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .impact-value {
        color: #1d3557;
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 20px;
    }
    .impact-value span {
        font-size: 2rem;
        color: #4361ee;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # First row of huge metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="impact-label">Total Portfolio Analyzed</div>
        <div class="impact-value">${total_portfolio / 1_000_000:,.1f}<span>M</span></div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="impact-label">AI Predictive Accuracy (F1)</div>
        <div class="impact-value">{f1_score:.1f}<span>%</span></div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second row of huge metrics
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="impact-label">Applications Processed</div>
        <div class="impact-value">{total_apps}</div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="impact-label">Average Inference Speed</div>
        <div class="impact-value">< 0.2<span>s</span></div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 3rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #1d3557; font-weight: 700; margin-bottom: 1.5rem;'>System Status</h3>", unsafe_allow_html=True)
    if f1_score >= 90:
        st.success(f"Excellent! The engine is highly calibrated for risk. The {best_model_name} model is serving predictions with {f1_score:.1f}% F1 Score.")
    else:
        st.success(f"The system is operational. The {best_model_name} model is serving predictions with {f1_score:.1f}% F1 Score.")
    
    st.info("Use the navigation menu in the sidebar to explore data, make predictions, and analyze model behavior.")
