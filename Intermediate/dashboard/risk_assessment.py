import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

def render_risk_assessment():
    st.markdown("<h1 style='font-weight: 800;'>Risk Assessment Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 1.1rem;'>In-depth view of portfolio and individual financial health indicators.</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    if 'last_prediction' in st.session_state:
        st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Individual Applicant Risk Profile</h3>", unsafe_allow_html=True)
        res = st.session_state['last_prediction']
        inp = st.session_state['last_input']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Gauge Chart for Risk Score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['risk_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Applicant Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(0,0,0,0.2)"},
                    'steps': [
                        {'range': [0, 25], 'color': "#2a9d8f"}, # Low
                        {'range': [25, 50], 'color': "#e9c46a"}, # Medium
                        {'range': [50, 75], 'color': "#f4a261"}, # High
                        {'range': [75, 100], 'color': "#e63946"} # Very High
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': res['risk_score']}
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### Financial Health Indicators")
            
            # Derived financial health
            total_income = inp['ApplicantIncome'] + inp['CoapplicantIncome']
            loan_burden = inp['LoanAmount'] / (total_income + 1)
            
            st.metric("Total Monthly Income", f"${total_income:,.2f}")
            st.metric("Loan-to-Income Ratio", f"{loan_burden:.4f}", delta="Lower is better", delta_color="inverse")
            st.metric("Risk Category", res['risk_category'])
            
            if res['risk_category'] in ["High Risk", "Very High Risk"]:
                st.error("Applicant presents significant default risk. Manual review recommended.")
            else:
                st.success("Applicant risk profile is within acceptable limits.")
                
    else:
        st.info("Make a prediction in the **Loan Prediction** tab to see individual risk profiles here.")
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Portfolio Risk Distribution</h3>", unsafe_allow_html=True)
    with st.expander("Explore the Engine's Capabilities...", expanded=False):
        st.write("These charts compare the historical distribution of risk scores and categories across all previously analyzed applications.")
    
    log_file = os.path.join(os.path.dirname(__file__), '../reports/prediction_logs.csv')
    if os.path.exists(log_file):
        df_logs = pd.read_csv(log_file)
        if len(df_logs) > 0:
            col3, col4 = st.columns(2)
            with col3:
                fig_risk = px.histogram(df_logs, x='risk_score', nbins=20, title='Historical Risk Score Distribution', color_discrete_sequence=['#4361ee'])
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col4:
                risk_cats = df_logs['risk_category'].value_counts().reset_index()
                risk_cats.columns = ['Risk Category', 'Count']
                fig_pie = px.pie(risk_cats, names='Risk Category', values='Count', title="Portfolio Breakdown by Risk Category", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Not enough historical data to display portfolio risk distribution.")
    else:
        st.info("No prediction logs found yet. Make some predictions to see portfolio analytics.")
