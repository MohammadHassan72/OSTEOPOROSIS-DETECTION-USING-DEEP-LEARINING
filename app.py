import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
import cv2

# PDF
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Osteoporosis Detection Using Deep Learning",
    layout="wide",
    page_icon="🧬"
)

IMAGE_SIZE = 256
CLASS_NAMES = ['normal', 'osteoporosis']

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_trained_model():
    m = load_model('osteoporosis_model.h5' , compile=False)

    # Warmup
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    m.predict(dummy)

    return m

model = load_trained_model()

# =========================
# GRAD-CAM
# =========================
def make_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer = None

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

# =========================
# OVERLAY
# =========================
def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img.astype(np.uint8), 0.7, heatmap, 0.3, 0)
    return overlay

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
.title {font-size:2.5em;font-weight:bold;color:#38bdf8;text-align:center;}
.card {background:#0f172a;padding:20px;border-radius:15px;margin-top:20px;color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧬 Osteoporosis Detection AI</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📋 Patient Details")
p_name = st.sidebar.text_input("Patient Name", "Guest")
p_age = st.sidebar.number_input("Age", 1, 100, 40)
p_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
p_id = st.sidebar.text_input("Report ID", "DX-2026-01")

# =========================
# PREPROCESS
# =========================
def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    arr = img_to_array(img)
    return np.expand_dims(arr, axis=0)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    processed_image = preprocess_image(uploaded_file)

    preds = model.predict(processed_image)
    pred_index = np.argmax(preds[0])
    confidence = preds[0][pred_index]
    predicted_label = CLASS_NAMES[pred_index].capitalize()

    # Heatmap
    heatmap = make_gradcam_heatmap(processed_image, model)
    overlay = overlay_heatmap(image_np, heatmap)

    with col2:
        st.image(overlay, caption="AI Heatmap", use_container_width=True)

    # Result card
    st.markdown(f"""
    <div class="card">
    <h4>Prediction</h4>
    Condition: {predicted_label}<br>
    Confidence: {confidence:.2%}
    </div>
    """, unsafe_allow_html=True)

    # Chart
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0])
    st.pyplot(fig)

    # =========================
    # PDF GENERATION
    # =========================
    if st.button("📥 Generate PDF Report"):

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # HEADER
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawCentredString(width/2, 760, "BONE DENSITY AI REPORT")

        pdf.setFont("Helvetica", 11)
        pdf.drawCentredString(width/2, 742, "Binary Classification: Normal vs Osteoporosis")
        pdf.line(50, 730, width-50, 730)

        # PATIENT INFO
        pdf.setFont("Helvetica", 11)
        pdf.drawString(60, 700, f"Patient Name: {p_name}")
        pdf.drawString(320, 700, f"Report ID: {p_id}")

        pdf.drawString(60, 680, f"Age: {p_age}")
        pdf.drawString(320, 680, f"Gender: {p_gender}")

        pdf.drawString(60, 660, f"Date: {datetime.now().strftime('%d-%b-%Y')}")
        pdf.drawString(320, 660, f"Technique: CNN Screening")

        # DIAGNOSTIC
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, 620, "DIAGNOSTIC SUMMARY")

        clinical = "Healthy Bone Density" if predicted_label == "Normal" else "High Fracture Risk Detected"
        reliability = "High" if confidence > 0.75 else "Moderate"

        data = [
            ["PARAMETER", "VALUE", "CLINICAL SIGNIFICANCE"],
            ["AI Classification", predicted_label, clinical],
            ["Score (0.0 - 1.0)", f"{confidence:.4f}", "Screening Result"],
            ["Reliability", reliability, "Manual Review Recommended"]
        ]

        table = Table(data, colWidths=[150,150,200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E5E7EB")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.grey)
        ]))

        table.wrapOn(pdf, width, height)
        table.drawOn(pdf, 50, 460)

        # IMAGE FRAME
        pdf.rect(120, 210, 350, 230)

        heatmap_img = Image.fromarray(overlay)
        if heatmap_img.mode != "RGB":
            heatmap_img = heatmap_img.convert("RGB")

        pdf.drawInlineImage(heatmap_img, 130, 220, width=330, height=210)

        # IMPRESSION
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, 180, "Clinical Impression:")

        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, 165, f"The system detects '{predicted_label}'.")

        # FOOTER
        pdf.setFont("Helvetica-Oblique", 8)
        pdf.drawCentredString(
            width/2, 80,
            "Disclaimer: This AI tool is for educational purposes only."
        ) 

        pdf.save()
        buffer.seek(0)

        st.download_button(
            "⬇ Download Professional Report",
            buffer,
            file_name=f"Report_{p_name}.pdf",
            mime="application/pdf"
        )

# st.markdown("---")
# st.caption("⚠️ AI tool for assistance only")
