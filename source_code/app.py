import streamlit as st
import os
from PIL import Image
import torch
import utils

# Page Configuration
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embedded Futuristic CSS
st.markdown("""
<style>
/* Main App Background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff;
    text-align: center;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-shadow: 0 0 15px rgba(88, 166, 255, 0.3);
}

/* Centered Button */
.stButton {
    display: flex;
    justify-content: center;
}

.stButton>button {
    background-color: #238636;
    color: white;
    border: 1px solid #2ea043;
    border-radius: 6px;
    padding: 10px 40px;
    font-size: 1.1em;
    box-shadow: 0 0 15px rgba(46, 160, 67, 0.4);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #2ea043;
    box-shadow: 0 0 25px rgba(46, 160, 67, 0.6);
    transform: scale(1.05);
}

/* Progress Bars */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #58a6ff, #8b949e);
}

/* Image Containers */
.element-container img {
    border: 1px solid #30363d;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    transition: transform 0.3s ease;
}

.element-container img:hover {
    transform: scale(1.01);
}

/* Result Card */
.result-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.prediction-title {
    font-size: 28px;
    font-weight: bold;
    color: #f78166;
    margin-bottom: 5px;
}

.prediction-score {
    font-size: 18px;
    color: #8b949e;
    font-family: 'Consolas', monospace;
}

/* Horizontal Rule with Text */
.separator {
    display: flex;
    align-items: center;
    text-align: center;
    color: #8b949e;
    margin: 30px 0;
}

.separator::before, .separator::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #30363d;
}

.separator:not(:empty)::before {
    margin-right: .5em;
}

.separator:not(:empty)::after {
    margin-left: .5em;
}
</style>
""", unsafe_allow_html=True)

# Interface Header
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3em; margin-bottom: 0;">🧠 NeuroScan AI</h1>
        <p style="font-size: 1.2em; color: #8b949e; margin-top: 10px;">
            Advanced Neural Interface for Brain Tumor Classification
        </p>
    </div>
    <hr style="border-color: #30363d;">
""", unsafe_allow_html=True)

# Sidebar - Model Selection & Settings
with st.sidebar:
    st.markdown("## ⚙️ System Configuration")
    
    model_choice = st.selectbox(
        "Select Neural Network Architecture",
        ("EfficientNet-B0", "ResNet18")
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    if model_choice == "EfficientNet-B0":
        st.info("**EfficientNet-B0**\n\nOptimized for efficiency and accuracy. Uses compound scaling method.")
    else:
        st.info("**ResNet18**\n\nClassic residual network architecture. Deep but lightweight.")
        
    st.markdown("---")
    st.markdown("### ℹ️ Classes")
    st.code("0: Glioma\n1: Meningioma\n2: No Tumor\n3: Pituitary")

# Main Content Area
# Top-down layout
st.markdown("<div class='separator'>START DIAGNOSTIC SEQUENCE</div>", unsafe_allow_html=True)

# 1. Upload Section (Centered)
col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])

with col_main:
    uploaded_file = st.file_uploader("Select Neural Scan (MRI Image)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 2. Preview Section
    with col_main:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Source Input", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("INITIATE ANALYSIS PROTOCOL", use_container_width=True)

    if analyze_btn:
        st.markdown("<div class='separator'>PROCESSING DATA</div>", unsafe_allow_html=True)
        
        with st.spinner("Accessing Neural Network Weights..."):
            # Load Model
            model_file = f"{model_choice.lower().replace('-', '_')}_brain_tumor.pth"
            # Point to the Models subdirectory relative to app.py
            model_path = os.path.join(os.path.dirname(__file__), 'Models', model_file)
            
            if not os.path.exists(model_path):
                st.error(f"System Failure: Model file missing at {model_path}")
            else:
                model = utils.load_model(model_choice, model_path)
                
                if model:
                    # Prediction
                    pred_class, confidence, probs = utils.predict_image(model, image)
                    
                    # 3. Results Section
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="prediction-title">{pred_class.upper()} DETECTED</div>
                            <div class="prediction-score">CONFIDENCE INTERVAL: {confidence*100:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 4. Visualizations (Side-by-Side)
                    st.markdown("### 🧬 Visual Diagnostics")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.markdown("**Class Probability Structure**")
                        for i, cls in enumerate(utils.CLASSES):
                            st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")
                            
                    with viz_col2:
                         st.markdown("**Attention Heatmap (Grad-CAM)**")
                         try:
                            heatmap_img = utils.generate_gradcam(model, image, model_choice)
                            if heatmap_img is not None:
                                overlay = utils.overlay_heatmap(image, heatmap_img)
                                st.image(overlay, caption="Model Focus Region", use_container_width=True)
                            else:
                                st.warning("Visual attention mapping unavailable for this architecture.")
                         except Exception as e:
                            st.error(f"Visualization Error: {e}")
                            
                    st.success("Analysis Cycle Complete.")

else:
    # Placeholder when no image
    with col_main:
        st.markdown("""
            <div style="text-align: center; color: #30363d; margin-top: 50px;">
                <h3 style="color: #30363d; text-shadow: none;">Awaiting Data Input</h3>
                <p>Upload MRI scan to activate predictive systems.</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; color: #484f58; font-size: 0.8em;">
        NeuroScan AI Interface v1.0 | Developed by Antigravity
    </div>
""", unsafe_allow_html=True)
