# Import required libraries
import streamlit as st
from streamlit_option_menu import option_menu
import os
import glob
import shutil
import pandas as pd
import time
from datetime import datetime
from PIL import Image
import PROCESS_X_RAY 
import PROCESS_CT_SCAN 
import DEPLOY_X_RAY 
import DEPLOY_CT_SCAN 
import DEPLOY_SYMPTOM 
from streamlit.components.v1 import html

def load_css():
    # Inject CSS for custom UI
    st.markdown("""
    <style>
    :root {
        --primary-blue: #0066cc;
        --primary-dark: #004c99;
        --secondary-gray: #f8f9fa;
        --border-gray: #dee2e6;
        --text-primary: #212529;
        --text-secondary: #6c757d;
        --success-green: #28a745;
        --warning-orange: #fd7e14;
        --danger-red: #dc3545;
        --clinical-blue: #e7f3ff;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background-color: #ffffff;
        padding: 0.5rem 1.5rem;
    }
    
    .professional-card {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .professional-card:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--primary-blue);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stDateInput label, .stTextArea label {
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select,
    .stDateInput input, .stTextArea textarea {
        border: 1px solid var(--border-gray) !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, 
    .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
    }
    
    .upload-section-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 12px;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 500;
        margin: 8px 0;
    }
    
    .status-pending {
        background: #f8f9fa;
        color: var(--text-secondary);
        border: 1px solid var(--border-gray);
    }
    
    .status-complete {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-processing {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .stButton > button {
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton > button[kind="primary"] {
        background-color: var(--primary-blue);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 2px 8px rgba(0,102,204,0.3);
    }
    
    .stButton > button:disabled {
        background-color: #c8c8c8 !important;
        color: #7a7a7a !important;
        cursor: not-allowed !important;
        opacity: 0.65 !important;
        box-shadow: none !important;
        border: 1px solid #b0b0b0 !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #6c757d;
        color: white;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #5a6268;
    }
    
    .stProgress > div > div > div > div {
        background-color: #9b59b6;
    }
    
    .progress-container {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 4px;
        margin: 16px 0;
    }
    
    .progress-text {
        font-size: 13px;
        color: var(--text-primary);
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .image-preview-container {
        border: 1px solid var(--border-gray);
        border-radius: 4px;
        padding: 8px;
        background: #ffffff;
        margin: 12px 0;
    }
    
    .alert {
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 14px;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #d4edda;
        color: #155724;
        border-color: #28a745;
    }
    
    .alert-warning {
        background: #fff3cd;
        color: #856404;
        border-color: #ffc107;
    }
    
    .alert-danger {
        background: #f8d7da;
        color: #721c24;
        border-color: #dc3545;
    }
    
    .alert-info {
        background: var(--clinical-blue);
        color: #004085;
        border-color: #0066cc;
    }
    
    .report-header {
        background: linear-gradient(to right, var(--primary-blue), var(--primary-dark));
        color: white;
        padding: 24px;
        border-radius: 4px 4px 0 0;
        margin-bottom: 0;
    }
    
    .report-body {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-top: none;
        padding: 24px;
        border-radius: 0 0 4px 4px;
    }
    
    .report-section {
        margin-bottom: 32px;
        padding-bottom: 24px;
        border-bottom: 1px solid var(--border-gray);
    }
    
    .report-section:last-child {
        border-bottom: none;
    }
    
    .report-section-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 16px;
    }
    
    .clinical-data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
    }
    
    .clinical-data-table th {
        background: #f8f9fa;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        border-bottom: 2px solid var(--border-gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .clinical-data-table td {
        padding: 12px;
        border-bottom: 1px solid var(--border-gray);
        font-size: 14px;
    }
    
    .clinical-divider {
        height: 1px;
        background: var(--border-gray);
        margin: 16px 0;
    }
    
    .required::after {
        content: " *";
        color: var(--danger-red);
        font-weight: bold;
    }
    
    button:focus, input:focus, textarea:focus, select:focus {
        outline: 3px solid rgba(0,102,204,0.3) !important;
        outline-offset: 2px !important;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    section[data-testid="stSidebar"] {
        background: #f8f9fa;
        border-right: 1px solid var(--border-gray);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    mark {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 2px;
    }
    
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    [data-testid="stFileUploader"] section {
        border: 2px dashed var(--border-gray);
        border-radius: 6px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--primary-blue);
        background: var(--clinical-blue);
    }
    </style>
    """, unsafe_allow_html=True)

def cleanup_directories():

    directories = [
        './Pre_Input_X-ray/*', './Input_X-ray/*', 
        './Pre_Input_CT-Scan/*', './Input_CT-Scan/*',
        './XAI_Output_1/*', './XAI_Output_2/*'
    ]
    for pattern in directories:
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

def ensure_directories():

    for dir_ in ['./Pre_Input_X-ray', './Input_X-ray', 
                 './Pre_Input_CT-Scan', './Input_CT-Scan',
                 './XAI_Output_1', './XAI_Output_2']:
        os.makedirs(dir_, exist_ok=True)

def stream_text(text, speed=0.02):

    for word in text.split(" "):
        yield word + " "
        time.sleep(speed)


def process_and_predict():

    st.session_state.analyzing = True
    
    # Create progress display
    st.markdown('<div class="professional-card" style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Analysis In Progress</div>', unsafe_allow_html=True)
    
    progress = st.progress(0)
    status = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    results = {}
    
    progress_steps = [
        (5, "Initializing analysis pipeline..."),
        (15, "Loading X-ray data..."),
        (25, "Processing X-ray imaging..."),
        (35, "Analyzing X-ray features..."),
        (45, "Loading CT scan data..."),
        (55, "Processing CT scan imaging..."),
        (65, "Analyzing CT scan features..."),
        (75, "Extracting clinical features..."),
        (85, "Running AI models..."),
        (95, "Integrating multimodal results..."),
        (100, "Analysis complete!")
    ]
    
    step_idx = 0
    
    # Initialization
    status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
    progress.progress(progress_steps[step_idx][0] / 100)
    step_idx += 1
    
    # Process X-ray
    if os.path.exists('./Pre_Input_X-ray') and os.listdir('./Pre_Input_X-ray'):
        # Loading
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        step_idx += 1
        
        # Processing
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        PROCESS_X_RAY.process()
        step_idx += 1
        
        # Analyzing
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        DEPLOY_X_RAY.predict()
        results['xray_results'] = DEPLOY_X_RAY.prediction_results
        step_idx += 1
    else:
        # Skip X-ray steps if no data
        step_idx += 3
    
    # Process CT Scan
    if os.path.exists('./Pre_Input_CT-Scan') and os.listdir('./Pre_Input_CT-Scan'):
        # Loading
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        step_idx += 1
        
        # Processing
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        PROCESS_CT_SCAN.process()
        step_idx += 1
        
        # Analyzing
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        DEPLOY_CT_SCAN.predict()
        results['ct_results'] = DEPLOY_CT_SCAN.prediction_results
        step_idx += 1
    else:
        # Skip CT steps if no data
        step_idx += 3
    
    # Process Clinical Notes
    if st.session_state.get('clinical_text'):
        status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
        progress.progress(progress_steps[step_idx][0] / 100)
        DEPLOY_SYMPTOM.Clinical_Note = st.session_state.clinical_text
        DEPLOY_SYMPTOM.predict()
        results['symptom_results'] = DEPLOY_SYMPTOM.prediction_results
        step_idx += 1
    else:
        step_idx += 1
    
    status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
    progress.progress(progress_steps[step_idx][0] / 100)
    step_idx += 1
    
    # Integration
    status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #212529;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
    progress.progress(progress_steps[step_idx][0] / 100)
    integrated_results = integrate_prediction_results(
        results.get('ct_results', {}),
        results.get('xray_results', {}),
        results.get('symptom_results', {})
    )
    step_idx += 1
    
    # Complete
    status.markdown(f'<p class="progress-text" style="font-size: 14px; font-weight: 500; color: #28a745;">{progress_steps[step_idx][1]}</p>', unsafe_allow_html=True)
    progress.progress(progress_steps[step_idx][0] / 100)
    
    # Clear flags and store results
    st.session_state.button_clicked = False
    st.session_state.analyzing = False  
    time.sleep(0.8) 
    st.session_state.results = integrated_results
    st.rerun()
    return integrated_results

# Define symptom weights
SYMPTOM_RISK_WEIGHTS = {
    'High Risk': 12.0,
    'Moderate Risk': 8.0,
    'Low Risk': 5.0
}

def calculate_base_weights(ct_class: str, x_class: str, risk_level: str) -> dict:
    
    # Define CT cancer types that increase CT weight
    ct_cancer_types = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma'] 
    
    # Give 50% weight if CT result matches known cancer types
    ct_weight = 50 if ct_class in ct_cancer_types else 0 
    
    # Give 40% weight if X-ray detects a nodule
    x_weight = 40 if x_class == 'Nodule' else 0 

    risk_weight = SYMPTOM_RISK_WEIGHTS.get(risk_level, 5.0)
    
    weights = {
        'ct_weight': ct_weight,
        'xray_weight': x_weight,
        'symptom_weight': risk_weight,
        'total': ct_weight + x_weight + risk_weight
    }
    
    print("-"*25)
    print(f"[DEBUG] CT Weight: {ct_weight}%")
    print(f"[DEBUG] X-ray Weight: {x_weight}%")
    print(f"[DEBUG] Symptom Weight: {risk_weight}%")
    print(f"[DEBUG] Total Base Weight: {weights['total']}%")

    return weights


def calculate_confidence_adjustments(ct_confidence: float, x_confidence: float, 
                                    s_confidence: float, risk_level: str) -> dict:
    
    symptom_weight = SYMPTOM_RISK_WEIGHTS.get(risk_level, 5.0)
    
    # Set baseline weights for each modality
    baseline_weights = {
        'CT': 50,
        'X-ray': 40,
        'Symptoms': symptom_weight 
    }
    
    confidence_list = [
        {'modality': 'CT', 'confidence': ct_confidence},
        {'modality': 'X-ray', 'confidence': x_confidence}, 
        {'modality': 'Symptoms', 'confidence': s_confidence} 
    ] 
    
    # Sort modalities from highest to lowest confidence
    sorted_confidences = sorted(confidence_list, key=lambda x: x['confidence'], reverse=True)
    
    # Apply proportional adjustments based on confidence ranking 
    adjustments = {
        'highest': {
            'modality': sorted_confidences[0]['modality'],
            'adjustment': baseline_weights[sorted_confidences[0]['modality']] * 0.08
        },
        'middle': {
            'modality': sorted_confidences[1]['modality'],
            'adjustment': baseline_weights[sorted_confidences[1]['modality']] * 0.04
        },
        'lowest': {
            'modality': sorted_confidences[2]['modality'],
            'adjustment': baseline_weights[sorted_confidences[2]['modality']] * 0.02
        }
    }
    
    total_adjustment = (adjustments['highest']['adjustment'] + 
                       adjustments['middle']['adjustment'] + 
                       adjustments['lowest']['adjustment'])
    
    adjustments['total'] = total_adjustment
    
    return adjustments


def calculate_overall_cancer_suspicion(results):
    
    # Extract predicted classes from CT and X-ray
    ct_pred = results.get('ct_prediction', '').strip()
    x_pred = results.get('xray_prediction', '').strip()
    
    clinical_results = results.get('clinical_results', {})
    risk_level = clinical_results.get('risk_level', 'Low Risk')
    
    # Extract model confidence values for each modality
    ct_confidence = results.get('ct_confidence', 0)
    x_confidence = results.get('xray_confidence', 0)
    s_confidence = clinical_results.get('s_confidence', 0) 
    
    risk_weight = calculate_base_weights(ct_pred, x_pred, risk_level)['total']
    
    # Compute additional adjustment based on confidence rankings
    confidence_adjustment = calculate_confidence_adjustments(
        ct_confidence, x_confidence, s_confidence, risk_level
    )['total']
    
    total_score = risk_weight + confidence_adjustment
    
    # Flag case as suspicious if total exceeds threshold (100%)
    is_suspicious = total_score > 100
    
    # Determine text label for final assessment
    suspicion_description = (
        "Findings Suggestive of Early Lung Malignancy"
        if is_suspicious
        else "No Definite Evidence of Malignancy"
    )
    
    return is_suspicious, suspicion_description, total_score


def integrate_prediction_results(ct_results: dict, x_results: dict, s_results: dict) -> dict:

    integrated_results = {}
    
    if ct_results:
        integrated_results.update({
            'ct_prediction': ct_results.get('ct_prediction', ''),
            'ct_confidence': ct_results.get('ct_confidence', 0)
        })
    
    if x_results:
        integrated_results.update({
            'xray_prediction': x_results.get('xray_prediction', ''),
            'xray_confidence': x_results.get('xray_confidence', 0)
        })
    
    if s_results and 'clinical_results' in s_results:
        integrated_results['clinical_results'] = s_results['clinical_results']
    
    is_suspicious, suspicion_description, adjusted_confidence = calculate_overall_cancer_suspicion(integrated_results)
    
    integrated_results.update({
        'is_suspicious': is_suspicious,
        'suspicion_description': suspicion_description,
        'adjusted_confidence': adjusted_confidence
    })

    return integrated_results

def display_medical_report_header(patient_name, results, stream_speed=0.1):
    
    # Clinical Assessment
    is_suspicious = results.get('is_suspicious', False)
    suspicion_description = results.get('suspicion_description', 'Results Pending')
    adjusted_confidence = results.get('adjusted_confidence', 0)
    
    if is_suspicious:
        st.markdown(f'<div class="alert alert-danger"><strong>Clinical Assessment:</strong> {suspicion_description} (Confidence: {adjusted_confidence:.2f}%)</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert alert-warning"><strong>Clinical Assessment:</strong> {suspicion_description} (Confidence: {adjusted_confidence:.2f}%)</div>', 
                   unsafe_allow_html=True)
    
def display_xray_section(results, stream_speed=0.02):

    if 'xray_prediction' not in results:
        return
        
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown('<div class="report-section-title">1. X-ray Analysis</div>', unsafe_allow_html=True)
    
    xray_text1 = f"""
    Radiographic analysis utilizing deep learning algorithms has identified findings consistent with **{results['xray_prediction']}** 
    with a confidence level of {results['xray_confidence']:.2f}%. Explainable AI techniques have been applied to generate a visual 
    interpretation overlay, wherein green-highlighted regions indicate areas contributing positively to the diagnostic assessment, 
    while red-highlighted regions represent areas that reduce diagnostic confidence. The visualization represents the top 5 
    superpixel regions of clinical significance.
    """
    
    xray_text2 = f"""Green-demarcated regions demonstrate radiographic features characteristic of **{results['xray_prediction']}**, 
    including specific parenchymal patterns, density variations, or structural abnormalities that correlate with the predicted diagnosis. 
    The extent and intensity of green highlighting correlate directly with the model's diagnostic certainty. Conversely, 
    red-demarcated regions exhibit features that introduce diagnostic uncertainty, potentially representing areas with atypical 
    presentation, artifact interference, or findings inconsistent with the primary diagnostic impression, thereby attenuating 
    the overall confidence score.
    """
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write_stream(stream_text(xray_text1, stream_speed))
        st.write_stream(stream_text(xray_text2, stream_speed))
    with col2:
        if os.path.exists('./XAI_Output_1/X-ray_LIME_Overlay.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            image = Image.open('./XAI_Output_1/X-ray_LIME_Overlay.png')
            st.image(
                image,
                caption="LIME Superpixel Attribution Map Highlighting Key X-ray Regions",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('./XAI_Output_1/X-ray_LIME_Bar.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(
                './XAI_Output_1/X-ray_LIME_Bar.png',
                caption="Regional Feature Importance Distribution for X-ray Classification",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if os.path.exists('./XAI_Output_1/X-ray_CNN_Influence.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(
                './XAI_Output_1/X-ray_CNN_Influence.png',
                caption="Ensemble Model Component Contributions in X-ray Decision",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_ct_section(results, stream_speed=0.02):

    if 'ct_prediction' not in results:
        return
        
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown('<div class="report-section-title">2. CT scan Analysis</div>', unsafe_allow_html=True)
    
    ct_text1 = f"""Cross-sectional imaging analysis has been performed utilizing convolutional neural network architectures, 
    yielding a diagnostic impression of **{results['ct_prediction']}** with {results['ct_confidence']:.2f}% confidence. 
    This confidence metric reflects the algorithmic certainty of the classification, with elevated values indicating robust 
    correlation between observed imaging features and known pathological patterns. A gradient-weighted class activation heatmap 
    has been generated to delineate anatomical regions of highest diagnostic relevance, employing a color-gradient schema to 
    quantify regional contribution to the final classification.
    """
    
    ct_text2 = f"""Regions demonstrating increased thermal intensity (red-yellow spectrum) represent anatomical areas exhibiting 
    imaging characteristics most strongly associated with **{results['ct_prediction']}**, indicating focal areas where the 
    algorithm detected significant pathognomonic features. This activation mapping provides spatial localization of the primary 
    diagnostic findings, enabling correlation between algorithmic interpretation and anatomical structures. The visualization 
    serves to elucidate the computational decision-making process and highlights regions warranting focused clinical attention.
    """
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write_stream(stream_text(ct_text1, stream_speed))
        st.write_stream(stream_text(ct_text2, stream_speed))
    with col2:
        if os.path.exists('./XAI_Output_2/CT_GradCAM++.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            image = Image.open('./XAI_Output_2/CT_GradCAM++.png')
            st.image(
                image,
                caption="Grad-CAM++ Heatmap of High-Impact CT Regions",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('./XAI_Output_2/CT_LIME_Overlay.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(
                './XAI_Output_2/CT_LIME_Overlay.png',
                caption="LIME Superpixel Attribution Map Highlighting Key CT Regions",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if os.path.exists('./XAI_Output_2/CT_LIME_Bar.png'):
            st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
            st.image(
                './XAI_Output_2/CT_LIME_Bar.png',
                caption="Regional Feature Importance Distribution for CT Classification",
                use_column_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_clinical_section(results, stream_speed=0.07):

    if 'clinical_results' not in results:
        return
        
    clinical_data = results['clinical_results']
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown('<div class="report-section-title">3. Clinical Symptom Analysis</div>', unsafe_allow_html=True)

    symptoms_list = [f"**{s['head_symptom']}**" for s in clinical_data['symptoms']]
    
    symptom_text = f"""
    Natural language processing analysis of the clinical documentation has identified pertinent symptomatology, 
    including {', '.join(symptoms_list)}. Each identified symptom has been assigned a clinical weight based on 
    its documented association with pulmonary pathology. The aggregate symptom burden has been assessed to 
    determine overall risk stratification, enabling systematic evaluation of diagnostic probability.
    """

    st.write_stream(stream_text(symptom_text, stream_speed))
    
    # Risk Assessment Table
    risk_html = '<table class="clinical-data-table"><tr><th>Risk Parameter</th><th>Value</th></tr>'
    risk_html += f'<tr><td>Risk Stratification</td><td><strong>{clinical_data["risk_level"]}</strong></td></tr>'
    risk_html += f'<tr><td>Risk Weighting Factor</td><td>{clinical_data["risk_weight"]}%</td></tr>'
    risk_html += '</table>'
    st.markdown(risk_html, unsafe_allow_html=True)
    
    # Expandable Clinical Note Annotation
    with st.expander("AI-Extracted Symptom Mentions in Clinical Documentation"):
        clinical_text = st.session_state.clinical_text
        final_text = clinical_text
        
        highlights = []
        for symptom in clinical_data['symptoms']:
            highlights.append({
                'start': symptom['start'],
                'end': symptom['end'],
                'text': symptom['text'],
                'head_symptom': symptom['head_symptom']
            })
        
        highlights.sort(key=lambda x: x['start'], reverse=True)
        
        for highlight in highlights:
            before = final_text[:highlight['start']]
            after = final_text[highlight['end']:]
            highlighted = final_text[highlight['start']:highlight['end']]
            final_text = before + f"<mark title='{highlight['head_symptom']}'>{highlighted}</mark>" + after
        
        st.markdown(final_text, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_medical_report(results, patient_name, stream_speed=0.02):

    display_medical_report_header(patient_name, results, stream_speed=0.1)
    display_xray_section(results, stream_speed)
    display_ct_section(results, stream_speed)
    display_clinical_section(results, stream_speed=0.07)

def run():

    if 'initialized' not in st.session_state:
        cleanup_directories()
        ensure_directories()
        
        st.session_state.step = 'upload'
        st.session_state.step1 = 'No'
        st.session_state.final2 = False
        st.session_state.final3 = False
        st.session_state.final4 = False
        st.session_state.initialized = True
        st.session_state.run_analysis_clicked = False
        st.session_state.analyzing = False  # Flag to prevent interruptions
        st.session_state.final = False
        st.session_state.results = None
        st.session_state.patient_name = None
        st.session_state.uploaded_xray = None
        st.session_state.uploaded_ct = None
        st.session_state.patient_age = None
        st.session_state.patient_gender = None
        st.session_state.patient_mrn = None
        st.session_state.exam_date = None
        st.session_state.physician = None

def upload():

    load_css()
    
    # Check if analysis is running
    is_analyzing = st.session_state.get('analyzing', False)
    is_locked = st.session_state.get('run_analysis_clicked', False) or is_analyzing
    
    st.title("M³LungXAI Clinical Diagnostic System")
    st.markdown("AI-Assisted Multimodal Early Lung Cancer Detection")
    
    # Show analysis in progress message if analyzing
    if is_analyzing:
        st.markdown('<div class="alert alert-info">Analysis in progress. Please wait...</div>', 
                   unsafe_allow_html=True)
    
    # Patient Information Section
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Patient Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_name = st.text_input(
            "Patient Name",
            value=st.session_state.get('patient_name', ''),
            placeholder="Last, First Middle",
            key="patient_name_input",
            disabled=is_locked,
            help="Required field"
        )
        if not is_locked:
            st.session_state.patient_name = patient_name
        
        patient_age = st.text_input(
            "Age (years)",
            value=str(st.session_state.get('patient_age', '')) if st.session_state.get('patient_age') else '',
            placeholder="e.g., 67",
            key="patient_age_input",
            disabled=is_locked
        )
        if patient_age and not is_locked:
            st.session_state.patient_age = patient_age
    
    with col2:
        patient_gender = st.selectbox(
            "Gender",
            options=["Not Specified", "Male", "Female", "Other"],
            index=0,
            key="patient_gender_input",
            disabled=is_locked
        )
        if patient_gender != "Not Specified" and not is_locked:
            st.session_state.patient_gender = patient_gender
        
        patient_mrn = st.text_input(
            "Medical Record Number",
            value=st.session_state.get('patient_mrn', ''),
            placeholder="MRN-XXXX-XXXX",
            key="patient_mrn_input",
            disabled=is_locked
        )
        if patient_mrn and not is_locked:
            st.session_state.patient_mrn = patient_mrn
    
    with col3:
        exam_date = st.date_input(
            "Examination Date",
            value=datetime.now().date(),
            key="exam_date_input",
            disabled=is_locked
        )
        if not is_locked:
            st.session_state.exam_date = exam_date
        
        physician = st.text_input(
            "Referring Physician",
            value=st.session_state.get('physician', ''),
            placeholder="Dr. Last Name",
            key="physician_input",
            disabled=is_locked
        )
        if physician and not is_locked:
            st.session_state.physician = physician
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Medical Data Upload Section
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Medical Imaging & Clinical Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    # X-ray Upload
    with col1:
        st.markdown('<p style="font-weight:600; margin-bottom:12px;">X-ray Imaging</p>', unsafe_allow_html=True)
        
        if not is_locked:
            xray_file = st.file_uploader(
                "Upload X-ray",
                type=['png', 'jpg', 'jpeg'],
                key="xray",
                help="Supported: PNG, JPG, JPEG",
                label_visibility="collapsed"
            )
            if xray_file:
                st.session_state.uploaded_xray = xray_file
                image = Image.open(xray_file)
                st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                st.image(image.resize((300, 300)), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.uploaded_xray:
                image = Image.open(st.session_state.uploaded_xray)
                st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                st.image(image.resize((300, 300)), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # CT Scan Upload
    with col2:
        st.markdown('<p style="font-weight:600; margin-bottom:12px;">CT scan Imaging</p>', unsafe_allow_html=True)
        
        if not is_locked:
            ct_file = st.file_uploader(
                "Upload CT scan",
                type=['png', 'jpg', 'jpeg'],
                key="ct",
                help="Supported: PNG, JPG, JPEG",
                label_visibility="collapsed"
            )
            if ct_file:
                st.session_state.uploaded_ct = ct_file
                image = Image.open(ct_file)
                st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                st.image(image.resize((300, 300)), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.uploaded_ct:
                image = Image.open(st.session_state.uploaded_ct)
                st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
                st.image(image.resize((300, 300)), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Clinical Notes
    with col3:
        st.markdown('<p style="font-weight:600; margin-bottom:12px;">Clinical Notes</p>', unsafe_allow_html=True)
        
        # Get the current value from session state
        current_clinical_text = st.session_state.get('clinical_text', '')
        
        clinical_text = st.text_area(
            "Clinical Notes",
            value=current_clinical_text,
            placeholder="Enter clinical observations, symptoms, medical history...",
            key="clinical_notes",
            height=300,
            disabled=is_locked,
            label_visibility="collapsed"
        )
        
        # Only update session state if not locked
        if not is_locked:
            if clinical_text:
                st.session_state.clinical_text = clinical_text
        
        if current_clinical_text or clinical_text:
            display_text = current_clinical_text if is_locked else clinical_text
            word_count = len(display_text.split())
            st.caption(f"{word_count} words")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action Buttons
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    col1, col2, col3 = st.columns([3, 4, 3])
    
    with col2:
        # Disable button if analyzing or already clicked
        run_button = st.button(
            "Run Analysis",
            disabled=is_analyzing or st.session_state.get('run_analysis_clicked', False),
            use_container_width=True,
            type="primary"
        )
    
    # Only process button click if not already analyzing
    if run_button and not is_analyzing:
            missing = []
            if not patient_name:
                missing.append("Patient Name")
            if not st.session_state.get('uploaded_xray') and not st.session_state.get('uploaded_ct') and not clinical_text:
                missing.append("At least one data source")
            
            if missing:
                st.markdown(f'<div class="alert alert-danger">Missing required fields: {", ".join(missing)}</div>', 
                           unsafe_allow_html=True)
            else:
                st.session_state.button_clicked = True
                st.session_state.run_analysis_clicked = True
                st.rerun()

    # Only process if button was clicked and not currently analyzing
    if st.session_state.button_clicked and not is_analyzing:
        cleanup_directories()
        
        # Save clinical text before processing if it exists
        if clinical_text and not st.session_state.get('clinical_text'):
            st.session_state.clinical_text = clinical_text

        if st.session_state.uploaded_xray:
            with open(os.path.join('./Pre_Input_X-ray', st.session_state.uploaded_xray.name), 'wb') as f:
                f.write(st.session_state.uploaded_xray.getbuffer())
        if st.session_state.uploaded_ct:
            with open(os.path.join('./Pre_Input_CT-Scan', st.session_state.uploaded_ct.name), 'wb') as f:
                f.write(st.session_state.uploaded_ct.getbuffer())

        results_data = process_and_predict()
        st.session_state.results = results_data
        st.session_state.step = 'upload'
        st.session_state.button_clicked = False
        st.session_state.run_analysis_clicked = False
        st.rerun()

    if st.session_state.get('results') is not None and not is_analyzing:
        st.markdown('<div class="alert alert-success">Analysis complete. Navigate to Results tab to view report.</div>', 
               unsafe_allow_html=True)


def results():

    load_css()
    
    # Check if analysis is running
    is_analyzing = st.session_state.get('analyzing', False)
    
    st.title("Medical Analysis Report")
    
    # Display patient information
    if st.session_state.results is not None:
        display_name = st.session_state.patient_name if st.session_state.patient_name else "Anonymous"
        
        # Patient Information Table
        patient_info_html = '<table class="clinical-data-table"><tr><th>Field</th><th>Value</th></tr>'
        patient_info_html += f'<tr><td>Patient Name</td><td>{display_name}</td></tr>'
        
        if st.session_state.get('patient_age'):
            patient_info_html += f'<tr><td>Age</td><td>{st.session_state.patient_age} years</td></tr>'
        if st.session_state.get('patient_gender'):
            patient_info_html += f'<tr><td>Gender</td><td>{st.session_state.patient_gender}</td></tr>'
        if st.session_state.get('patient_mrn'):
            patient_info_html += f'<tr><td>Medical Record Number</td><td>{st.session_state.patient_mrn}</td></tr>'
        
        if st.session_state.get('exam_date'):
            exam_date_str = st.session_state.exam_date.strftime("%B %d, %Y")
            patient_info_html += f'<tr><td>Examination Date</td><td>{exam_date_str}</td></tr>'
        else:
            patient_info_html += f'<tr><td>Examination Date</td><td>{datetime.now().strftime("%B %d, %Y at %H:%M")}</td></tr>'
        
        if st.session_state.get('physician'):
            patient_info_html += f'<tr><td>Referring Physician</td><td>{st.session_state.physician}</td></tr>'
        
        patient_info_html += '</table>'
        st.markdown(patient_info_html, unsafe_allow_html=True)
        
        st.markdown('<div class="clinical-divider"></div>', unsafe_allow_html=True)
    
    # Generate Report button
    col1, col2, col3 = st.columns([3, 4, 3])
    with col2:
        if not st.session_state.get("final", False):
            if st.button("Generate Report", use_container_width=True, type="primary", disabled=False):
                st.session_state.step1 = 'results'
                st.session_state.final = True
                st.rerun()
        elif st.session_state.get("final2", False):
            st.button("Generate Report", use_container_width=True, disabled=False)
            st.session_state.step1 = 'results_stay'
        else:
            st.button("Generate Report", use_container_width=True, disabled=False)

    if st.session_state.step1 == 'results':
        if st.session_state.results is not None:
            display_medical_report(st.session_state.results, st.session_state.patient_name, stream_speed=0.02)
            st.session_state.final2 = True
        else:
            st.markdown('<div class="alert alert-danger">No analysis results available. Please run analysis from the Data Entry tab.</div>', 
                       unsafe_allow_html=True)
    
    if st.session_state.step1 == 'results_stay':
        if st.session_state.results is not None:
            display_medical_report(st.session_state.results, st.session_state.patient_name, stream_speed=0.0)
            st.session_state.final2 = True
        else:
            st.markdown('<div class="alert alert-danger">No analysis results available. Please run analysis from the Data Entry tab.</div>', 
                       unsafe_allow_html=True)

    if st.session_state.step1 == 'No':
        st.markdown('<div class="alert alert-info">Click "Generate Report" to view detailed analysis results.</div>', 
                   unsafe_allow_html=True)

st.set_page_config(
    page_title="M³LungXAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main navigation
selected1 = option_menu(
    menu_title=None,
    options=["Data Entry", "Results"],
    icons=["clipboard-data", "file-medical"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0", "background-color": "#f8f9fa", "border-bottom": "2px solid #dee2e6"},
        "icon": {"font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0",
            "padding": "16px",
            "color": "#0066cc",
            "font-weight": "500"
        },
        "nav-link-selected": {
            "background-color": "#0066cc",
            "color": "white",
            "font-weight": "600"
        }
    },
)

run()

if selected1 == "Results":
    results()

if selected1 == "Data Entry":
    upload()
