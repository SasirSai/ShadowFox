import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from prediction import RiskAssessmentEngine

def render_prediction():
    st.markdown("<h1 style='font-weight: 600;margin-top: 0px;'>Loan Applicant Profiler</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 1.1rem;'>Adjust applicant parameters below to see the real-time AI Risk Assessment.</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: 700; margin-bottom: 1rem;'>Applicant Profile Setup</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("<div style='color: #1d3557; font-weight: 700; margin-bottom: 10px;'>Demographics</div>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female"], help="The gender of the primary applicant.")
            married = st.selectbox("Married", ["No", "Yes"], help="Current marital status of the primary applicant.")
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], help="Number of individuals depending on the applicant's income.")
        
    with col2:
        with st.container(border=True):
            st.markdown("<div style='color: #1d3557; font-weight: 700; margin-bottom: 10px;'>Professional Profile</div>", unsafe_allow_html=True)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], help="Highest level of completed education.")
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], help="Does the applicant own their own business or are they a freelancer?")
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="The location type of the property being purchased.")
        
    with col3:
        with st.container(border=True):
            st.markdown("<div style='color: #1d3557; font-weight: 700; margin-bottom: 10px;'>Financial Setup</div>", unsafe_allow_html=True)
            applicant_income = st.slider("Applicant Income ($)", min_value=150, max_value=81000, value=5000, step=100, help="Gross monthly income of the primary applicant.")
            coapplicant_income = st.slider("Coapplicant Income ($)", min_value=0, max_value=42000, value=0, step=100, help="Gross monthly income of the co-applicant (if applicable).")
            loan_amount = st.slider("Loan Amount ($k)", min_value=9.0, max_value=700.0, value=150.0, step=5.0, help="The total loan amount requested in thousands.")
            loan_amount_term = st.selectbox("Loan Term (Months)", [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0], index=8, help="The duration over which the loan will be repaid.")
            credit_history = st.selectbox("Credit History", ["Good", "Poor"], help="Whether the applicant has a good credit history (meets guidelines) or poor credit history.")
            credit_history_val = 1.0 if credit_history == "Good" else 0.0
        
    # Prepare input for real-time prediction
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history_val,
        "Property_Area": property_area
    }
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Risk Score", type="primary", use_container_width=True)
    
    if predict_clicked:
        with st.spinner("Analyzing applicant profile..."):
            engine = RiskAssessmentEngine()
            result, X_processed = engine.predict(input_dict)
            
            # Store in session state for other pages
            st.session_state['last_prediction'] = result
            st.session_state['last_input'] = input_dict
            st.session_state['last_X_processed'] = X_processed
                
        st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Real-Time Prediction Results</h3>", unsafe_allow_html=True)
        
        # Display Results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        status_color = "#2a9d8f" if result['prediction'] == "Approved" else "#e63946"
        
        with res_col1:
            st.markdown(f"""
            <div style="text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0;">
                <h4 style="color: #6c757d; margin-bottom: 5px; font-size: 0.9rem; text-transform: uppercase;">AI Decision</h4>
                <h1 style="color: {status_color}; margin: 0; font-size: 2.5rem;">{result['prediction']}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div style="text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0;">
                <h4 style="color: #6c757d; margin-bottom: 5px; font-size: 0.9rem; text-transform: uppercase;">Confidence</h4>
                <h1 style="color: #1d3557; margin: 0; font-size: 2.5rem;">{float(result['confidence']):.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
        risk_color = "#2a9d8f" if result['risk_category'] == "Low Risk" else "#f4a261" if result['risk_category'] == "Medium Risk" else "#e63946"
        with res_col3:
            st.markdown(f"""
            <div style="text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0;">
                <h4 style="color: #6c757d; margin-bottom: 5px; font-size: 0.9rem; text-transform: uppercase;">Risk Score</h4>
                <h1 style="color: {risk_color}; margin: 0; font-size: 2.5rem;">{float(result['risk_score']):.1f} <span style='font-size: 1.2rem; color: #6c757d'>/ 100</span></h1>
            </div>
            """, unsafe_allow_html=True)
            
        st.info("💡 Adjust the sliders above and click Predict again to see how income or loan amount impacts the decision instantly. Navigate to **Risk Assessment** or **Explainability (SHAP)** to dive deeper.")
