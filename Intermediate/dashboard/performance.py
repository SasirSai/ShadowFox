import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_performance(metrics, best_model_name):
    st.markdown("<h1 style='font-weight: 800;'>Model Performance & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 1.1rem;'>Comparative analysis of machine learning models trained on the portfolio.</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Best Model: {}</h3>".format(best_model_name), unsafe_allow_html=True)
    
    # Convert metrics dictionary to DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
    df_metrics.rename(columns={'index': 'Model'}, inplace=True)
    
    # Best Model Deep Dive
    best_metrics = metrics.get(best_model_name, {})
    bm_col1, bm_col2, bm_col3, bm_col4 = st.columns(4)
    with bm_col1:
        st.metric("Test Accuracy", f"{best_metrics.get('Accuracy', 0)*100:.1f}%")
    with bm_col2:
        st.metric("Precision", f"{best_metrics.get('Precision', 0)*100:.1f}%")
    with bm_col3:
        st.metric("Recall", f"{best_metrics.get('Recall', 0)*100:.1f}%")
    with bm_col4:
        st.metric("F1 Score", f"{best_metrics.get('F1 Score', 0)*100:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: 700; margin-bottom: 1rem;'>Performance Metrics Overview</h3>", unsafe_allow_html=True)
    st.dataframe(df_metrics.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], color='#d4edda', axis=0), use_container_width=True)
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Plotly Bar Chart
    st.markdown("<h3 style='font-weight: 700; margin-bottom: 1.5rem;'>Model Benchmarking</h3>", unsafe_allow_html=True)
    with st.expander("Explore the Engine's Capabilities...", expanded=False):
        st.write("A visual comparison of how each model performed across standard classification metrics. Higher bars indicate better performance.")
    metrics_melted = df_metrics.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], var_name='Metric', value_name='Score')
    fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group', title='Model Comparison')
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("<hr style='border: none; height: 1px; background-color: #e0e0e0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='color: #1d3557; font-weight: 700; margin-bottom: 1rem;'>Best Performing Model</h3>", unsafe_allow_html=True)
        st.success(f"**{best_model_name}** was selected as the production model due to its optimal balance of performance metrics.")
        
        st.info("""
        **Evaluation Criteria:**
        - **Accuracy:** Overall correctness.
        - **Precision:** Focuses on minimizing False Positives (incorrect approvals).
        - **Recall:** Focuses on minimizing False Negatives (incorrect rejections).
        - **F1 Score:** Harmonic mean of Precision and Recall.
        - **ROC-AUC:** Ability to distinguish between classes.
        """)
        
    with col2:
        # Radar chart for Best Model
        best_metrics = df_metrics[df_metrics['Model'] == best_model_name].iloc[0]
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[best_metrics[c] for c in categories],
            theta=categories,
            fill='toself',
            name=best_model_name,
            line_color='#4361ee'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title=f"{best_model_name} Performance Signature",
            height=350,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
