import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from explainability import SHAPExplainer

def render_explainability():
    st.markdown("<h1 style='font-weight: 800;'>Explainable AI (SHAP Analysis)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 1.1rem;'>Transparent decision logic for every loan assessment.</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    with st.spinner("Loading Explainability Engine..."):
        explainer = SHAPExplainer()
        
    if 'last_prediction' in st.session_state:
        st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Individual Prediction Explanation</h3>", unsafe_allow_html=True)
        
        res = st.session_state['last_prediction']
        X_processed = st.session_state['last_X_processed']
        
        st.info(explainer.get_human_readable_explanation(X_processed, res['prediction'], res['confidence']))
        
        st.markdown("#### SHAP Waterfall Plot")
        with st.expander("Explore the Engine's Capabilities..."):
            st.write("The waterfall plot explains how each feature contributed to the final prediction. Starting from the baseline (average), red arrows push the risk higher, and blue arrows push the risk lower, leading to the final decision.")
        
        fig = explainer.get_waterfall_plot(X_processed)
        if fig:
            st.pyplot(fig)
        else:
            st.error("Waterfall plot not available for this model type.")
    else:
        st.info("Make a prediction in the **Loan Prediction** tab first.")
