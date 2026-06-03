import streamlit as st
import pandas as pd
import json
import os
import joblib
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(
    page_title="Fintech Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism Sidebar and Clean Main Area
st.markdown("""
<style>
    /* Main area background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Deep Dark Sidebar - BMW Theme */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: none !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Headers with Vibrant Styling */
    h1, h2 {
        color: #1e3c72 !important; /* Fallback for browsers that don't support text-fill */
        background: linear-gradient(45deg, #1e3c72, #2a5298, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
        padding-bottom: 10px;
    }
    h3 {
        color: #2a5298 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-bottom: 1.5rem;
    }
    
    /* Hide text ('key') inside the collapse button by shrinking font and making it transparent */
    [data-testid="stSidebarCollapseButton"], 
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] kbd {
        color: transparent !important;
        font-size: 0px !important;
    }
    
    /* Ensure the sidebar chevron icon is visible and bright blue */
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #4361ee !important;
        stroke: #4361ee !important;
        width: 1.5rem;
        height: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Import pages
from dashboard.home import render_home
from dashboard.eda import render_eda
from dashboard.prediction import render_prediction
from dashboard.risk_assessment import render_risk_assessment
from dashboard.explainability import render_explainability
from dashboard.performance import render_performance

def main():
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4361ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("Risk Intelligence")
    st.sidebar.markdown("---")
    
    # Premium Vertical Navigation Menu (BMW Style)
    with st.sidebar:
        choice = option_menu(
            menu_title="",
            options=[
                "Home",
                "Data Analysis (EDA)",
                "Loan Prediction",
                "Risk Assessment",
                "Explainability (SHAP)",
                "Model Performance"
            ],
            icons=['house', 'bar-chart', 'calculator', 'shield-check', 'lightbulb', 'graph-up'],
            default_index=0,
            key="main_menu_unique",
            styles={
                "container": {"padding": "0!important", "background-color": "#000000", "border-radius": "0px"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px 0px", "--hover-color": "#1c1c1c", "color": "white", "font-weight": "600", "border-radius": "5px"},
                "nav-link-selected": {"background-color": "#0066B1"},
            }
        )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Enterprise AI Credit Risk Engine v1.0")
    
    # Load common data
    data_path = os.path.join(os.path.dirname(__file__), 'data/loan_prediction.csv')
    df = pd.read_csv(data_path)
    
    metrics_path = os.path.join(os.path.dirname(__file__), 'reports/metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    best_model_name = joblib.load(os.path.join(os.path.dirname(__file__), 'models/best_model_name.pkl'))
        
    # Route to pages
    if choice == "Home":
        render_home(df, metrics, best_model_name)
    elif choice == "Data Analysis (EDA)":
        render_eda(df)
    elif choice == "Loan Prediction":
        render_prediction()
    elif choice == "Risk Assessment":
        render_risk_assessment()
    elif choice == "Explainability (SHAP)":
        render_explainability()
    elif choice == "Model Performance":
        render_performance(metrics, best_model_name)

if __name__ == '__main__':
    main()
