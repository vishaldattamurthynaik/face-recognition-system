"""
FaceMatrix: Face Recognition & Identification System.
Built with Streamlit, OpenCV, and ONNX Runtime.
"""
import io
import time
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_L2_THRESHOLD,
    DATABASE_PATH,
    ENROLLED_DIR,
    ARTIFACTS_DIR,
    UNKNOWN_LABEL
)
from src.pipeline import FaceRecognitionPipeline
from evaluation.benchmark import run_full_benchmark


# Page Configuration
st.set_page_config(
    page_title="FaceMatrix - Biometric Face Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Definitions
THEMES = {
    "Indigo Obsidian": {
        "bg_base": "#090A10",
        "bg_card": "#121324",
        "bg_card_hover": "#1A1B33",
        "border_color": "rgba(99, 102, 241, 0.2)",
        "border_hover": "rgba(129, 140, 248, 0.6)",
        "accent": "#818CF8",
        "accent_known": "#10B981",
        "accent_unknown": "#F43F5E",
        "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8",
    },
    "Emerald Matrix": {
        "bg_base": "#050F0A",
        "bg_card": "#0C1E15",
        "bg_card_hover": "#132D20",
        "border_color": "rgba(16, 185, 129, 0.22)",
        "border_hover": "rgba(52, 211, 153, 0.6)",
        "accent": "#34D399",
        "accent_known": "#10B981",
        "accent_unknown": "#F87171",
        "text_primary": "#F0FDF4",
        "text_secondary": "#86EFAC",
    },
    "Amber Sunset": {
        "bg_base": "#0C0A09",
        "bg_card": "#1C1917",
        "bg_card_hover": "#2B2623",
        "border_color": "rgba(245, 158, 11, 0.22)",
        "border_hover": "rgba(251, 191, 36, 0.6)",
        "accent": "#FBBF24",
        "accent_known": "#10B981",
        "accent_unknown": "#EF4444",
        "text_primary": "#FAFAF9",
        "text_secondary": "#A8A29E",
    },
    "Cyber Teal": {
        "bg_base": "#040D14",
        "bg_card": "#0A1C29",
        "bg_card_hover": "#102C40",
        "border_color": "rgba(20, 184, 166, 0.22)",
        "border_hover": "rgba(45, 212, 191, 0.6)",
        "accent": "#2DD4BF",
        "accent_known": "#10B981",
        "accent_unknown": "#F43F5E",
        "text_primary": "#F0FDFA",
        "text_secondary": "#99F6E4",
    }
}


@st.cache_resource
def load_pipeline():
    """
    Initializes and caches the FaceRecognitionPipeline.
    """
    return FaceRecognitionPipeline()


pipeline = load_pipeline()


# Sidebar Controls
st.sidebar.markdown("### FaceMatrix")
st.sidebar.caption("Biometric Identification System")
st.sidebar.markdown("---")

st.sidebar.markdown("#### Color Palette")
selected_theme_name = st.sidebar.selectbox(
    "Theme",
    options=list(THEMES.keys()),
    index=0,
    key="app_theme_select",
    label_visibility="collapsed"
)
t = THEMES[selected_theme_name]

st.sidebar.markdown("#### Matching Settings")
sim_threshold = st.sidebar.slider(
    "Cosine Threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(DEFAULT_COSINE_THRESHOLD),
    step=0.02,
    help="Decision boundary for open-set identification. Higher values reject more impostors."
)
pipeline.set_threshold(sim_threshold)

st.sidebar.markdown("#### Display Options")
show_landmarks = st.sidebar.checkbox("Draw Facial Landmarks", value=True)
show_sim_tags = st.sidebar.checkbox("Show Match Badges", value=True)

st.sidebar.markdown("---")
db_count = pipeline.database.count()
st.sidebar.metric("Enrolled Database", f"{db_count} profiles")

st.sidebar.markdown("---")
st.sidebar.caption("Engine: OpenCV YuNet + SFace (ONNX)\nTarget: Local CPU/DNN")


# Clean CSS Injection (Zero markdown indentation to avoid raw text leaks)
custom_css = f"""
<style>
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulseDot {{
    0% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
    100% {{ opacity: 1; }}
}}

.stApp {{
    background-color: {t['bg_base']} !important;
    color: {t['text_primary']} !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    animation: fadeInUp 0.3s ease-out;
}}

div[data-testid="stMetric"] {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border_color']};
    border-radius: 10px;
    padding: 14px 18px;
    transition: all 0.25s ease;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {t['border_hover']};
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: {t['accent']} !important;
}}
div[data-testid="stMetricLabel"] {{
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {t['text_secondary']} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {t['border_color']};
    padding-bottom: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.92rem;
    font-weight: 500;
    color: {t['text_secondary']};
    transition: all 0.2s ease;
    border: 1px solid transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {t['text_primary']};
    background-color: rgba(255, 255, 255, 0.05);
}}
.stTabs [aria-selected="true"] {{
    background-color: {t['bg_card']} !important;
    color: {t['accent']} !important;
    border-color: {t['border_color']} !important;
}}

.stButton > button {{
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid {t['border_color']};
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    border-color: {t['border_hover']};
}}

.badge-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.8rem;
    font-weight: 600;
}}
.badge-known {{
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}}
.badge-unknown {{
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# Main Header
st.title("Face Recognition & Identification System")
st.caption("Deep metric face identification with 5-point alignment, centroid modeling, and open-set unknown rejection.")

# KPI Metrics Grid using native Streamlit columns
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cosine Threshold", f"{sim_threshold:.2f}", help="Decision Boundary")
k2.metric("Enrolled Identities", f"{db_count} profiles", help="Registered Subjects")
k3.metric("Embedding Space", "128-D", help="Hypersphere Centroids")
k4.metric("Inference Engine", "ONNX", help="Local CPU / DNN")

st.markdown("---")

# Navigation Tabs
tab_identify, tab_live, tab_enroll, tab_sandbox, tab_eval, tab_db = st.tabs([
    "Image Identification",
    "Webcam Identification",
    "Enroll Identity",
    "Threshold Calibration",
    "Evaluation & Benchmark",
    "Database Management"
])


# ==========================================
# TAB 1: IMAGE IDENTIFICATION
# ==========================================
with tab_identify:
    st.subheader("Query Image Identification")
    st.write("Upload an image containing one or multiple subjects for biometric detection and identification.")

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=["jpg", "jpeg", "png", "webp"],
        key="ident_upload"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            t0 = time.time()
            output = pipeline.recognize_faces(
                image,
                return_annotated=True,
                draw_landmarks=show_landmarks,
                show_similarity=show_sim_tags
            )
            proc_time_ms = (time.time() - t0) * 1000

            res_left, res_right = st.columns([1.1, 0.9], gap="large")

            with res_left:
                st.markdown("##### Annotated Visual Output")
                annotated_rgb = cv2.cvtColor(output.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(
                    annotated_rgb,
                    caption=f"Inference Latency: {proc_time_ms:.1f} ms | Detected Faces: {output.num_faces}",
                    use_container_width=True
                )

            with res_right:
                st.markdown("##### Match Breakdown")
                st.write(f"Total Detected: **{output.num_faces}** | Known: **{output.num_known}** | Unknown: **{output.num_unknown}**")

                if output.num_faces == 0:
                    st.info("No face detected. Please ensure clear lighting and front-facing orientation.")
                else:
                    for idx, (det, match) in enumerate(zip(output.detections, output.matches)):
                        box_title = f"Face #{idx+1}: {match.name} ({match.confidence_pct}%)"
                        with st.expander(box_title, expanded=(idx == 0)):
                            crop_col, metrics_col = st.columns([1, 2])
                            with crop_col:
                                crop = pipeline.detector.extract_face_crop(image, det)
                                if crop.size > 0:
                                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                            with metrics_col:
                                if match.is_known:
                                    st.success(f"Identified: {match.name}")
                                else:
                                    st.error(f"Rejected as Unknown (Similarity: {match.similarity_score:.3f} < {sim_threshold:.2f})")

                                st.write(f"- Cosine Similarity: `{match.similarity_score:.4f}`")
                                st.write(f"- Decision Threshold: `{sim_threshold:.2f}`")
                                st.write(f"- Detector Score: `{det.score:.3f}`")
                                st.write(f"- Margin to 2nd: `{match.margin:.4f}`")

                            if match.all_candidates:
                                st.caption("Top Candidate Similarities:")
                                cand_df = pd.DataFrame([
                                    {"Candidate": c.name, "Similarity": c.cosine_similarity}
                                    for c in match.all_candidates[:5]
                                ])
                                st.bar_chart(cand_df.set_index("Candidate"), height=160)


# ==========================================
# TAB 2: LIVE WEBCAM IDENTIFICATION
# ==========================================
with tab_live:
    st.subheader("Webcam Face Identification")
    st.write("Capture a live frame from your webcam for real-time identification.")

    cam_left, cam_right = st.columns([1, 1], gap="large")

    with cam_left:
        cam_image = st.camera_input("Capture Snapshot", key="webcam_ident", label_visibility="collapsed")

    with cam_right:
        if cam_image is not None:
            cam_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(cam_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                t0 = time.time()
                output = pipeline.recognize_faces(
                    frame,
                    return_annotated=True,
                    draw_landmarks=show_landmarks,
                    show_similarity=show_sim_tags
                )
                dt = (time.time() - t0) * 1000

                st.markdown("##### Identification Result")
                st.image(
                    cv2.cvtColor(output.annotated_image, cv2.COLOR_BGR2RGB),
                    caption=f"Latency: {dt:.1f} ms | Detected Faces: {output.num_faces}",
                    use_container_width=True
                )

                for match in output.matches:
                    if match.is_known:
                        st.success(f"Identified: {match.name} (Similarity: {match.similarity_score:.3f}, Confidence: {match.confidence_pct}%)")
                    else:
                        st.warning(f"Unknown Identity (Similarity: {match.similarity_score:.3f} < Threshold: {sim_threshold:.2f})")
        else:
            st.info("Click 'Take Photo' on the left to capture and verify face identity.")


# ==========================================
# TAB 3: IDENTITY ENROLLMENT
# ==========================================
with tab_enroll:
    st.subheader("Enroll New Identity")
    st.write("Register an identity with one or multiple photos to compute a high-precision centroid embedding.")

    with st.form("enroll_form", clear_on_submit=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            enroll_name = st.text_input("Full Name *", placeholder="e.g. Marie Curie")
            enroll_id = st.text_input("Identity ID / Code (Optional)", placeholder="e.g. ID-104")
        with fcol2:
            enroll_role = st.text_input("Role / Department", placeholder="e.g. Research")
            enroll_notes = st.text_input("Notes / Metadata", placeholder="e.g. Primary Desk")

        enroll_method = st.radio("Input Source", ["Upload Image Files", "Webcam Snapshot"], horizontal=True)

        uploaded_enroll_files = []
        webcam_enroll_file = None

        if enroll_method == "Upload Image Files":
            uploaded_enroll_files = st.file_uploader(
                "Upload 1 to 5 face photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="enroll_files"
            )
        else:
            webcam_enroll_file = st.camera_input("Capture Enrollment Photo", key="enroll_cam")

        submitted = st.form_submit_button("Complete Enrollment", type="primary", use_container_width=True)

        if submitted:
            if not enroll_name.strip():
                st.error("Please enter a valid Full Name.")
            else:
                images_to_process = []
                if enroll_method == "Upload Image Files" and uploaded_enroll_files:
                    for uf in uploaded_enroll_files:
                        fb = np.asarray(bytearray(uf.read()), dtype=np.uint8)
                        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                        if img is not None:
                            images_to_process.append(img)
                elif enroll_method == "Webcam Snapshot" and webcam_enroll_file is not None:
                    fb = np.asarray(bytearray(webcam_enroll_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                    if img is not None:
                        images_to_process.append(img)

                if not images_to_process:
                    st.error("Please provide at least one valid image photo.")
                else:
                    success_count = 0
                    last_person = None
                    for img in images_to_process:
                        ok, msg, person = pipeline.enroll_face(
                            image=img,
                            name=enroll_name.strip(),
                            person_id=enroll_id.strip() if enroll_id else None,
                            metadata={"role": enroll_role, "notes": enroll_notes}
                        )
                        if ok:
                            success_count += 1
                            last_person = person

                    if success_count > 0 and last_person:
                        st.success(f"Enrolled {last_person.name} with {last_person.num_samples} sample(s). Centroid updated.")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Face detection failed on the provided image(s). Please verify framing and lighting.")


# ==========================================
# TAB 4: THRESHOLD CALIBRATION
# ==========================================
with tab_sandbox:
    st.subheader("Interactive Threshold Calibration")
    st.write("Analyze how decision threshold adjustments impact the boundary between accepted matches and rejected impostors.")

    sb_col1, sb_col2 = st.columns([1, 1], gap="large")

    with sb_col1:
        sb_file = st.file_uploader("Upload test image for threshold analysis", type=["jpg", "png", "jpeg"], key="sandbox_upload")

        if sb_file is not None:
            fb = np.asarray(bytearray(sb_file.read()), dtype=np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)

            if img is not None:
                dets = pipeline.detector.detect(img)
                if dets:
                    emb = pipeline.embedder.extract_embedding(img, dets[0])
                    matrix, pids, names = pipeline.database.get_all_centroids_matrix()

                    if len(pids) > 0:
                        sims = np.dot(matrix, emb)
                        top_idx = int(np.argmax(sims))
                        best_name = names[top_idx]
                        best_sim = float(sims[top_idx])
                        is_accepted = best_sim >= sim_threshold

                        aligned = pipeline.embedder.align_face(img, dets[0])
                        st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), width=160, caption="5-Point Aligned Face Chip")

                        if is_accepted:
                            st.success(f"Classification: **{best_name}** (Accepted)\n\nSimilarity `{best_sim:.3f}` >= Threshold `{sim_threshold:.2f}`")
                        else:
                            st.error(f"Classification: **Unknown** (Rejected)\n\nSimilarity `{best_sim:.3f}` < Threshold `{sim_threshold:.2f}`")

                        with sb_col2:
                            st.markdown("##### Cosine Similarity vs Enrolled Identities")
                            chart_df = pd.DataFrame({
                                "Identity": names,
                                "Cosine Similarity": sims
                            }).set_index("Identity")
                            st.bar_chart(chart_df, height=280)
                    else:
                        st.warning("Database is empty. Please enroll identities first.")
                else:
                    st.error("No face detected in test image.")
    with sb_col2:
        if sb_file is None:
            st.info("Upload an image on the left to evaluate similarity against all enrolled profiles.")


# ==========================================
# TAB 5: EVALUATION & BENCHMARK
# ==========================================
with tab_eval:
    st.subheader("Evaluation & Performance Benchmark")
    st.write("Standardized evaluation suite testing verification accuracy, False Acceptance Rate (FAR), and False Rejection Rate (FRR).")

    eval_col1, eval_col2 = st.columns([1, 3])
    with eval_col1:
        if st.button("Run Full Benchmark", type="primary", use_container_width=True):
            with st.spinner("Evaluating known pairs and unknown impostor sets..."):
                results = run_full_benchmark(custom_threshold=sim_threshold)
                st.success("Benchmark completed.")

    roc_img = ARTIFACTS_DIR / "roc_curve.png"
    thresh_img = ARTIFACTS_DIR / "threshold_analysis.png"
    cm_img = ARTIFACTS_DIR / "confusion_matrix.png"

    if roc_img.exists() and thresh_img.exists():
        pcol1, pcol2 = st.columns(2, gap="large")
        with pcol1:
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
        with pcol2:
            st.image(str(thresh_img), caption="FAR vs FRR Operating Point Curve", use_container_width=True)

        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix at Operating Threshold", width=540)
    else:
        st.info("Click 'Run Full Benchmark' to execute the test suite and generate evaluation graphs.")


# ==========================================
# TAB 6: DATABASE MANAGEMENT
# ==========================================
with tab_db:
    st.subheader("Enrolled Profiles Database")
    people = pipeline.database.list_people()

    if not people:
        st.info("Database is empty. Use the Enroll Identity tab to register profiles.")
    else:
        st.caption(f"Total Enrolled Profiles: {len(people)}")

        for p in people:
            with st.container():
                dcol1, dcol2, dcol3 = st.columns([1, 4, 1.2])
                with dcol1:
                    chip_path = ENROLLED_DIR / p.person_id / "sample_1.jpg"
                    if chip_path.exists():
                        st.image(str(chip_path), width=75)
                    else:
                        st.text("[No Photo]")
                with dcol2:
                    st.markdown(f"**{p.name}** `({p.person_id})`")
                    st.write(f"Samples: `{p.num_samples}` | Centroid Norm: `{np.linalg.norm(p.centroid):.4f}`")
                    if p.metadata:
                        st.caption(f"Role: {p.metadata.get('role', '-')} | Notes: {p.metadata.get('notes', '-')}")
                with dcol3:
                    if st.button("Delete Profile", key=f"del_{p.person_id}", use_container_width=True):
                        pipeline.database.delete_person(p.person_id)
                        st.rerun()
                st.markdown("---")

        if st.button("Clear Entire Database", type="secondary"):
            pipeline.database.clear()
            st.success("Database cleared.")
            st.rerun()
