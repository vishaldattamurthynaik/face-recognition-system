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

# Theme Palette Definitions
THEMES = {
    "Indigo Obsidian": {
        "bg_base": "#090A10",
        "bg_card": "#121324",
        "bg_card_hover": "#1A1B33",
        "border_color": "rgba(99, 102, 241, 0.16)",
        "border_hover": "rgba(129, 140, 248, 0.55)",
        "accent": "#818CF8",
        "accent_secondary": "#A855F7",
        "accent_known": "#10B981",
        "accent_unknown": "#F43F5E",
        "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "header_grad": "linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(15, 12, 41, 0.8) 100%)",
        "pulse_glow": "rgba(129, 140, 248, 0.4)",
    },
    "Emerald Matrix": {
        "bg_base": "#050F0A",
        "bg_card": "#0C1E15",
        "bg_card_hover": "#132D20",
        "border_color": "rgba(16, 185, 129, 0.18)",
        "border_hover": "rgba(52, 211, 153, 0.55)",
        "accent": "#34D399",
        "accent_secondary": "#10B981",
        "accent_known": "#10B981",
        "accent_unknown": "#F87171",
        "text_primary": "#F0FDF4",
        "text_secondary": "#86EFAC",
        "header_grad": "linear-gradient(135deg, rgba(6, 78, 59, 0.6) 0%, rgba(2, 44, 34, 0.8) 100%)",
        "pulse_glow": "rgba(16, 185, 129, 0.4)",
    },
    "Amber Sunset": {
        "bg_base": "#0C0A09",
        "bg_card": "#1C1917",
        "bg_card_hover": "#2B2623",
        "border_color": "rgba(245, 158, 11, 0.18)",
        "border_hover": "rgba(251, 191, 36, 0.55)",
        "accent": "#FBBF24",
        "accent_secondary": "#F97316",
        "accent_known": "#10B981",
        "accent_unknown": "#EF4444",
        "text_primary": "#FAFAF9",
        "text_secondary": "#A8A29E",
        "header_grad": "linear-gradient(135deg, rgba(69, 26, 3, 0.6) 0%, rgba(28, 25, 23, 0.8) 100%)",
        "pulse_glow": "rgba(245, 158, 11, 0.4)",
    },
    "Cyber Teal": {
        "bg_base": "#040D14",
        "bg_card": "#0A1C29",
        "bg_card_hover": "#102C40",
        "border_color": "rgba(20, 184, 166, 0.18)",
        "border_hover": "rgba(45, 212, 191, 0.55)",
        "accent": "#2DD4BF",
        "accent_secondary": "#06B6D4",
        "accent_known": "#10B981",
        "accent_unknown": "#F43F5E",
        "text_primary": "#F0FDFA",
        "text_secondary": "#99F6E4",
        "header_grad": "linear-gradient(135deg, rgba(19, 78, 74, 0.6) 0%, rgba(8, 51, 68, 0.8) 100%)",
        "pulse_glow": "rgba(20, 184, 166, 0.4)",
    }
}


@st.cache_resource
def load_pipeline():
    """
    Initializes and caches the FaceRecognitionPipeline.
    """
    return FaceRecognitionPipeline()


pipeline = load_pipeline()


# Sidebar Controls & Settings
with st.sidebar:
    st.markdown("### FaceMatrix")
    st.caption("Biometric Identification System")
    st.markdown("---")

    st.markdown("#### Color Palette")
    selected_theme_name = st.selectbox("Select Theme", list(THEMES.keys()), index=0)
    theme = THEMES[selected_theme_name]

    st.markdown("#### Matching Settings")
    sim_threshold = st.slider(
        "Cosine Threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(DEFAULT_COSINE_THRESHOLD),
        step=0.02,
        help="Decision boundary for open-set identification. Higher values reject more impostors."
    )
    pipeline.set_threshold(sim_threshold)

    st.markdown("#### Display Options")
    show_landmarks = st.checkbox("Draw Facial Landmarks", value=True)
    show_sim_tags = st.checkbox("Show Match Badges", value=True)

    st.markdown("---")
    db_count = pipeline.database.count()
    st.metric("Enrolled Database", f"{db_count} profiles")

    st.markdown("---")
    st.caption("Engine: OpenCV YuNet + SFace (ONNX)\nTarget: Local CPU/DNN")


# Dynamic Themed Minimal CSS with Animations
st.markdown(f"""
<style>
    /* CSS Variables from Active Theme */
    :root {{
        --bg-base: {theme["bg_base"]};
        --bg-card: {theme["bg_card"]};
        --bg-card-hover: {theme["bg_card_hover"]};
        --border-color: {theme["border_color"]};
        --border-hover: {theme["border_hover"]};
        --accent: {theme["accent"]};
        --accent-sec: {theme["accent_secondary"]};
        --accent-known: {theme["accent_known"]};
        --accent-unknown: {theme["accent_unknown"]};
        --text-primary: {theme["text_primary"]};
        --text-secondary: {theme["text_secondary"]};
        --radius: 10px;
    }}

    /* Keyframe Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 {theme["pulse_glow"]}; }}
        70% {{ box-shadow: 0 0 0 8px rgba(0, 0, 0, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); }}
    }}

    /* Global Typography & Layout */
    .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg-base);
        color: var(--text-primary);
        animation: fadeInUp 0.35s ease-out;
    }}

    /* Header Container */
    .header-box {{
        background: {theme["header_grad"]};
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
        transition: border-color 0.3s ease;
    }}
    .header-box:hover {{
        border-color: var(--border-hover);
    }}

    /* Stat Cards */
    .stat-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 16px 20px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.45s ease-out;
    }}
    .stat-card:hover {{
        transform: translateY(-2px);
        border-color: var(--border-hover);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.92rem;
        font-weight: 500;
        color: var(--text-secondary);
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary);
        background-color: rgba(255, 255, 255, 0.05);
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--bg-card) !important;
        color: var(--accent) !important;
        border-color: var(--border-color) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}

    /* Status Badges */
    .badge-known {{
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
    }}
    .badge-unknown {{
        display: inline-flex;
        align-items: center;
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }}
    .status-dot-active {{
        background-color: var(--accent);
        animation: pulseGlow 2s infinite;
    }}

    /* Button Micro-interactions */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid var(--border-color);
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Profile Grid Card */
    .profile-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }}
    .profile-card:hover {{
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
        transform: translateY(-1px);
    }}
</style>
""", unsafe_allow_html=True)


# Top Header Section
st.markdown(f"""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em;">Face Recognition & Identification System</h1>
            <p style="margin: 4px 0 0 0; color: {theme['text_secondary']}; font-size: 0.92rem;">Deep metric face identification with 5-point alignment, centroid modeling, and open-set unknown rejection.</p>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="status-dot status-dot-active"></span>
            <span style="font-size: 0.85rem; font-weight: 500; color: {theme['text_secondary']};">System Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Metrics Grid
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 0.8rem; color: {theme['text_secondary']}; text-transform: uppercase; font-weight: 500;">Cosine Threshold</div>
        <div style="font-size: 1.6rem; font-weight: 600; color: {theme['accent']}; margin-top: 4px;">{sim_threshold:.2f}</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Decision Boundary</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 0.8rem; color: {theme['text_secondary']}; text-transform: uppercase; font-weight: 500;">Enrolled Identities</div>
        <div style="font-size: 1.6rem; font-weight: 600; color: {theme['accent_known']}; margin-top: 4px;">{db_count}</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Registered Subjects</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 0.8rem; color: {theme['text_secondary']}; text-transform: uppercase; font-weight: 500;">Embedding Space</div>
        <div style="font-size: 1.6rem; font-weight: 600; color: {theme['text_primary']}; margin-top: 4px;">128-D</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Hypersphere Centroids</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 0.8rem; color: {theme['text_secondary']}; text-transform: uppercase; font-weight: 500;">Inference Runtime</div>
        <div style="font-size: 1.6rem; font-weight: 600; color: {theme['text_primary']}; margin-top: 4px;">ONNX</div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Zero External API Calls</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# Main Tabs
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
    st.markdown("#### Query Image Identification")
    st.caption("Upload an image containing one or multiple subjects for biometric detection and identification.")

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

            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            res_left, res_right = st.columns([1.1, 0.9], gap="large")

            with res_left:
                st.markdown("##### Annotated Visual Output")
                annotated_rgb = cv2.cvtColor(output.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(
                    annotated_rgb,
                    caption=f"Inference Latency: {proc_time_ms:.1f} ms | Detected Faces: {output.num_faces}",
                    width="stretch"
                )

            with res_right:
                st.markdown("##### Match Breakdown")
                
                # Summary stat badges
                st.markdown(f"""
                <div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;">
                    <span class="badge-known">Known: {output.num_known}</span>
                    <span class="badge-unknown">Unknown: {output.num_unknown}</span>
                    <span style="background: rgba(255,255,255,0.06); padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; color: {theme['text_secondary']};">Total: {output.num_faces}</span>
                </div>
                """, unsafe_allow_html=True)

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
                                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width="stretch")
                            with metrics_col:
                                if match.is_known:
                                    st.markdown(f"<span class='badge-known'>Identified: {match.name}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<span class='badge-unknown'>Rejected (Unknown)</span>", unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="font-size: 0.85rem; color: {theme['text_secondary']}; margin-top: 8px; line-height: 1.6;">
                                    Cosine Similarity: <strong style="color: {theme['text_primary']};">{match.similarity_score:.4f}</strong><br>
                                    Decision Threshold: <strong style="color: {theme['text_primary']};">{sim_threshold:.2f}</strong><br>
                                    Detection Confidence: <strong style="color: {theme['text_primary']};">{det.score:.3f}</strong><br>
                                    Margin to 2nd Best: <strong style="color: {theme['text_primary']};">{match.margin:.4f}</strong>
                                </div>
                                """, unsafe_allow_html=True)

                            if match.all_candidates:
                                st.markdown(f"<div style='margin-top: 8px; font-size: 0.82rem; color: {theme['text_secondary']};'>Top Candidate Similarities:</div>", unsafe_allow_html=True)
                                cand_df = pd.DataFrame([
                                    {"Candidate": c.name, "Similarity": c.cosine_similarity}
                                    for c in match.all_candidates[:5]
                                ])
                                st.bar_chart(cand_df.set_index("Candidate"), height=160)


# ==========================================
# TAB 2: LIVE WEBCAM IDENTIFICATION
# ==========================================
with tab_live:
    st.markdown("#### Webcam Face Identification")
    st.caption("Capture a live frame from your webcam for real-time identification.")

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
                    width="stretch"
                )

                for match in output.matches:
                    if match.is_known:
                        st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px; margin-top: 10px;">
                            <div style="font-weight: 600; color: #34D399;">Identified: {match.name}</div>
                            <div style="font-size: 0.85rem; color: {theme['text_secondary']}; margin-top: 4px;">Similarity: {match.similarity_score:.3f} | Confidence: {match.confidence_pct}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px; margin-top: 10px;">
                            <div style="font-weight: 600; color: #F87171;">Unknown Identity</div>
                            <div style="font-size: 0.85rem; color: {theme['text_secondary']}; margin-top: 4px;">Similarity: {match.similarity_score:.3f} &lt; Threshold {sim_threshold:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Click 'Take Photo' on the left to capture and verify face identity.")


# ==========================================
# TAB 3: IDENTITY ENROLLMENT
# ==========================================
with tab_enroll:
    st.markdown("#### Enroll New Identity")
    st.caption("Register an identity with one or multiple photos to compute a high-precision centroid embedding.")

    with st.form("enroll_form", clear_on_submit=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            enroll_name = st.text_input("Full Name *", placeholder="e.g. Marie Curie")
            enroll_id = st.text_input("Identity ID / Code (Optional)", placeholder="e.g. ID-104")
        with fcol2:
            enroll_role = st.text_input("Role / Department", placeholder="e.g. Research")
            enroll_notes = st.text_input("Notes / Metadata", placeholder="e.g. Primary Desk")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
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

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
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
    st.markdown("#### Interactive Threshold Calibration")
    st.caption("Analyze how decision threshold adjustments impact the boundary between accepted matches and rejected impostors.")

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
                            st.markdown(f"""
                            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 14px; margin-top: 12px;">
                                <div style="font-weight: 600; color: #34D399; font-size: 1.05rem;">Classification: {best_name} (Accepted)</div>
                                <div style="font-size: 0.85rem; color: {theme['text_secondary']}; margin-top: 4px;">Similarity: <strong>{best_sim:.3f}</strong> &ge; Threshold: <strong>{sim_threshold:.2f}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 14px; margin-top: 12px;">
                                <div style="font-weight: 600; color: #F87171; font-size: 1.05rem;">Classification: Unknown (Rejected)</div>
                                <div style="font-size: 0.85rem; color: {theme['text_secondary']}; margin-top: 4px;">Similarity: <strong>{best_sim:.3f}</strong> &lt; Threshold: <strong>{sim_threshold:.2f}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)

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
    st.markdown("#### Evaluation & Performance Benchmark")
    st.caption("Standardized evaluation suite testing verification accuracy, False Acceptance Rate (FAR), and False Rejection Rate (FRR).")

    eval_col1, eval_col2 = st.columns([1, 3])
    with eval_col1:
        if st.button("Run Full Benchmark", type="primary", use_container_width=True):
            with st.spinner("Evaluating known pairs and unknown impostor sets..."):
                results = run_full_benchmark(custom_threshold=sim_threshold)
                st.success("Benchmark completed.")

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    roc_img = ARTIFACTS_DIR / "roc_curve.png"
    thresh_img = ARTIFACTS_DIR / "threshold_analysis.png"
    cm_img = ARTIFACTS_DIR / "confusion_matrix.png"

    if roc_img.exists() and thresh_img.exists():
        pcol1, pcol2 = st.columns(2, gap="large")
        with pcol1:
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC)", width="stretch")
        with pcol2:
            st.image(str(thresh_img), caption="FAR vs FRR Operating Point Curve", width="stretch")

        if cm_img.exists():
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            st.image(str(cm_img), caption="Confusion Matrix at Operating Threshold", width=540)
    else:
        st.info("Click 'Run Full Benchmark' to execute the test suite and generate evaluation graphs.")


# ==========================================
# TAB 6: DATABASE MANAGEMENT
# ==========================================
with tab_db:
    st.markdown("#### Enrolled Profiles Database")
    people = pipeline.database.list_people()

    if not people:
        st.info("Database is empty. Use the Enroll Identity tab to register profiles.")
    else:
        st.caption(f"Total Enrolled Profiles: {len(people)}")
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

        for p in people:
            with st.container():
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                dcol1, dcol2, dcol3 = st.columns([1, 4, 1.2])
                with dcol1:
                    chip_path = ENROLLED_DIR / p.person_id / "sample_1.jpg"
                    if chip_path.exists():
                        st.image(str(chip_path), width=75)
                    else:
                        st.markdown("<div style='color: #64748B; font-size: 0.8rem;'>No Photo</div>", unsafe_allow_html=True)
                with dcol2:
                    st.markdown(f"<strong style='font-size: 1.05rem;'>{p.name}</strong> <span style='color: #64748B; font-size: 0.85rem;'>({p.person_id})</span>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color: {theme['text_secondary']}; font-size: 0.85rem; margin-top: 4px;'>Samples: {p.num_samples} | Centroid Norm: {np.linalg.norm(p.centroid):.4f}</div>", unsafe_allow_html=True)
                    if p.metadata:
                        st.caption(f"Role: {p.metadata.get('role', '-')} | Notes: {p.metadata.get('notes', '-')}")
                with dcol3:
                    if st.button("Delete Profile", key=f"del_{p.person_id}", use_container_width=True):
                        pipeline.database.delete_person(p.person_id)
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Entire Database", type="secondary"):
            pipeline.database.clear()
            st.success("Database cleared.")
            st.rerun()
