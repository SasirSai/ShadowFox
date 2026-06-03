import streamlit as st
import pandas as pd
import plotly.express as px

def render_eda(df):
    st.markdown("<h1>Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d;'>Interactive exploration of the loan applicant dataset.</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.subheader("Loan Approval Distribution")
    with st.expander("Explore the Engine's Capabilities...", expanded=False):
        st.write("This chart illustrates the historical proportion of approved versus rejected loan applications within the dataset, identifying baseline portfolio acceptance rates.")
    fig1 = px.pie(df, names='Loan_Status', title="Approval vs Rejection", hole=0.4, color_discrete_sequence=['#2a9d8f', '#e63946'])
    st.plotly_chart(fig1, use_container_width=True)
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Distribution")
        with st.expander("Explore the Engine's Capabilities..."):
            st.write("Box-plots and histograms showing applicant income ranges separated by loan approval status to determine if higher income correlates with approval.")
        fig2 = px.histogram(df, x='ApplicantIncome', color='Loan_Status', marginal="box", title="Applicant Income by Loan Status", color_discrete_map={'Y': '#2a9d8f', 'N': '#e63946'})
        st.plotly_chart(fig2, use_container_width=True)
            
    with col2:
        st.subheader("Loan Amount Distribution")
        with st.expander("Explore the Engine's Capabilities..."):
            st.write("Distribution of loan amounts requested. Helps identify if large loan requests face higher rejection frequencies.")
        fig3 = px.histogram(df, x='LoanAmount', color='Loan_Status', marginal="box", title="Loan Amount by Loan Status", color_discrete_map={'Y': '#2a9d8f', 'N': '#e63946'})
        st.plotly_chart(fig3, use_container_width=True)
            
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.subheader("Credit History Analysis")
    with st.expander("Explore the Engine's Capabilities..."):
        st.write("Credit history is a pivotal feature. A score of 1.0 indicates good credit history, while 0.0 indicates poor history.")
    fig4 = px.histogram(df, x='Credit_History', color='Loan_Status', barmode='group', title="Approval Rate based on Credit History", color_discrete_map={'Y': '#2a9d8f', 'N': '#e63946'})
    fig4.update_layout(xaxis_type='category')
    st.plotly_chart(fig4, use_container_width=True)
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Education & Approval")
        with st.expander("Explore the Engine's Capabilities..."):
            st.write("Analyzes the impact of graduate versus non-graduate applicant status on final approval.")
        fig5 = px.histogram(df, x='Education', color='Loan_Status', barmode='group', title="Approval based on Education", color_discrete_map={'Y': '#2a9d8f', 'N': '#e63946'})
        st.plotly_chart(fig5, use_container_width=True)
            
    with col4:
        st.subheader("Gender & Marital Status")
        with st.expander("Explore the Engine's Capabilities..."):
            st.write("Explores the demographic distribution and its relationship with the bank's lending decisions.")
        fig_gender = px.histogram(df, x='Gender', color='Loan_Status', barmode='group', title="Approval based on Gender", color_discrete_map={'Y': '#2a9d8f', 'N': '#e63946'})
        st.plotly_chart(fig_gender, use_container_width=True)
            
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    with st.expander("Explore the Engine's Capabilities..."):
        st.write("Pearson correlation matrix for numerical features. Highlights multicollinearity and primary drivers of loan amount/income.")
    num_df = df.select_dtypes(include=['int64', 'float64'])
    corr = num_df.corr()
    fig6 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale='Blues')
    st.plotly_chart(fig6, use_container_width=True)
