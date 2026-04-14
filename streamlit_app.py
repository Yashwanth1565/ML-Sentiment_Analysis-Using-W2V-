"""
streamlit_app.py  --  Sentiment Analysis Frontend (Word2Vec | Advanced UI)
--------------------------------------------------------------------------
Run: streamlit run streamlit_app.py
Make sure FastAPI is running first: uvicorn app:app --reload
"""

import streamlit as st
import requests
import time
import pandas as pd
from collections import Counter
from datetime import datetime

API_URL = "http://localhost:8000"

LABEL_CONFIG = {
    "Positive": {"emoji": "😊", "color": "#00C896", "bg": "rgba(0,200,150,0.12)", "border": "#00C896"},
    "Neutral":  {"emoji": "😐", "color": "#F5A623", "bg": "rgba(245,166,35,0.12)",  "border": "#F5A623"},
    "Negative": {"emoji": "😠", "color": "#FF4D6D", "bg": "rgba(255,77,109,0.12)",  "border": "#FF4D6D"},
}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
}

/* ── Background ── */
.stApp {
    background: #0A0A0F;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,150,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,60,255,0.05) 0%, transparent 50%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0D14 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #aaa !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #fff !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}

/* ── Result pill ── */
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 1.4rem;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    border: 1.5px solid;
    margin: 0.8rem 0 0.4rem;
}

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label { font-size: 0.72rem; color: #666; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Batch result rows ── */
.batch-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.65rem 1rem;
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.02);
    font-size: 0.88rem;
    transition: all 0.2s;
}
.batch-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
    border: 1px solid;
}
.batch-text { color: #bbb; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Tags ── */
.meta-tag {
    display: inline-block;
    font-size: 0.72rem;
    color: #555;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 2px 8px;
    margin: 2px 3px;
}

/* ── Progress bar overrides ── */
.stProgress > div > div { background: #00C896 !important; border-radius: 4px; }

/* ── Button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #00C896, #00A87A) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.8rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #eee !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
}
.stTextArea textarea:focus {
    border-color: #00C896 !important;
    box-shadow: 0 0 0 2px rgba(0,200,150,0.15) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #666 !important;
    border-radius: 8px !important;
    padding: 0.4rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,200,150,0.15) !important;
    color: #00C896 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Success / error / warning ── */
.stAlert { border-radius: 10px !important; border: none !important; }

/* ── Selectbox / input ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #eee !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {text, label, time_ms, ts}
if "total_analysed" not in st.session_state:
    st.session_state.total_analysed = 0


# ── API helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return (True, r.json()) if r.status_code == 200 else (False, {})
    except Exception:
        return False, {}

def predict_single(text):
    resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
    resp.raise_for_status()
    return resp.json()

def predict_batch(texts):
    resp = requests.post(f"{API_URL}/predict/batch", json={"texts": texts}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 SentimentIQ")
    st.markdown('<p style="color:#555;font-size:0.8rem;margin-top:-8px;">Word2Vec · FastAPI · Streamlit</p>', unsafe_allow_html=True)
    st.divider()

    api_ok, api_info = check_api()

    if api_ok:
        st.markdown(f"""
        <div style="background:rgba(0,200,150,0.08);border:1px solid rgba(0,200,150,0.2);
                    border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;">
            <div style="color:#00C896;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;margin-bottom:8px;">● API ONLINE</div>
            <div style="color:#888;font-size:0.75rem;">Model: <span style="color:#ddd">{api_info.get('model','—')}</span></div>
            <div style="color:#888;font-size:0.75rem;">Vocab: <span style="color:#ddd">{api_info.get('w2v_vocab',0):,} words</span></div>
            <div style="color:#888;font-size:0.75rem;">Vectors: <span style="color:#ddd">{api_info.get('vector_size','—')}d</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("● API OFFLINE\nRun: uvicorn app:app --reload")
        st.stop()

    st.markdown("### Session Stats")
    total = st.session_state.total_analysed
    hist  = st.session_state.history
    pos   = sum(1 for h in hist if h["label"] == "Positive")
    neu   = sum(1 for h in hist if h["label"] == "Neutral")
    neg   = sum(1 for h in hist if h["label"] == "Negative")

    st.markdown(f"""
    <div class="stat-grid" style="flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="color:#666;">Total analysed</span>
            <span style="color:#eee;font-family:'Syne',sans-serif;font-weight:700;">{total}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:4px 0;">
            <span style="color:#00C896;">😊 Positive</span><span style="color:#eee;">{pos}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:4px 0;">
            <span style="color:#F5A623;">😐 Neutral</span><span style="color:#eee;">{neu}</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:4px 0;">
            <span style="color:#FF4D6D;">😠 Negative</span><span style="color:#eee;">{neg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if hist:
        st.divider()
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_analysed = 0
            st.rerun()

    st.divider()
    st.markdown('<p style="color:#333;font-size:0.72rem;text-align:center;">SentimentIQ v2.0 · Word2Vec</p>', unsafe_allow_html=True)


# ── Main header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
               color:#fff;letter-spacing:-0.02em;margin-bottom:4px;">
        🧬 Sentiment<span style="color:#00C896;">IQ</span>
    </h1>
    <p style="color:#555;font-size:0.88rem;margin:0;">
        Real-time sentiment classification · Word2Vec embeddings · 3-class output
    </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Single Analysis",
    "📦 Batch Analysis",
    "📊 Analytics",
    "🕐 History",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### Enter text to analyse")

        # Quick example buttons
        st.markdown('<p style="color:#555;font-size:0.78rem;margin-bottom:6px;">QUICK EXAMPLES</p>', unsafe_allow_html=True)
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        example_text = ""
        with ex_col1:
            if st.button("😊 Positive", use_container_width=True):
                example_text = "Absolutely love this product! Best purchase I've made all year."
        with ex_col2:
            if st.button("😐 Neutral", use_container_width=True):
                example_text = "The product arrived on time. It works as described."
        with ex_col3:
            if st.button("😠 Negative", use_container_width=True):
                example_text = "Terrible quality. Broke after two days. Complete waste of money."

        user_text = st.text_area(
            label="",
            value=example_text,
            placeholder="Type or paste any comment, review, or text here...",
            height=160,
            key="single_input",
            label_visibility="collapsed",
        )

        char_count = len(user_text) if user_text else 0
        st.markdown(
            f'<p style="color:#444;font-size:0.75rem;text-align:right;">{char_count}/5000 chars</p>',
            unsafe_allow_html=True,
        )

        analyse_btn = st.button("⚡ Analyse Sentiment", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Result")

        result_placeholder = st.empty()

        if analyse_btn:
            if not user_text or len(user_text.strip()) < 3:
                result_placeholder.warning("Please enter at least 3 characters.")
            else:
                with st.spinner(""):
                    try:
                        data  = predict_single(user_text)
                        label = data["label"]
                        cfg   = LABEL_CONFIG[label]

                        # Save to history
                        st.session_state.history.insert(0, {
                            "text"    : user_text[:120],
                            "label"   : label,
                            "time_ms" : data["time_ms"],
                            "ts"      : datetime.now().strftime("%H:%M:%S"),
                        })
                        st.session_state.total_analysed += 1

                        result_placeholder.markdown(f"""
                        <div class="card" style="border-color:{cfg['border']}22;">
                            <div style="color:#666;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">PREDICTION</div>
                            <div class="result-pill" style="color:{cfg['color']};background:{cfg['bg']};border-color:{cfg['border']};">
                                {cfg['emoji']} &nbsp;{label.upper()}
                            </div>
                            <div style="margin-top:1rem;">
                                <div style="color:#444;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">CLEANED INPUT</div>
                                <div style="color:#888;font-size:0.8rem;line-height:1.6;font-style:italic;">
                                    "{data['cleaned'][:180]}{'...' if len(data['cleaned'])>180 else ''}"
                                </div>
                            </div>
                            <div style="margin-top:1rem;display:flex;gap:8px;flex-wrap:wrap;">
                                <span class="meta-tag">⏱ {data['time_ms']} ms</span>
                                <span class="meta-tag">📏 {len(user_text)} chars</span>
                                <span class="meta-tag">🔤 {len(user_text.split())} words</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except requests.exceptions.ConnectionError:
                        result_placeholder.error("Lost connection to API.")
                    except Exception as e:
                        result_placeholder.error(f"Error: {e}")
        else:
            result_placeholder.markdown("""
            <div class="card" style="min-height:200px;display:flex;align-items:center;justify-content:center;">
                <div style="text-align:center;color:#333;">
                    <div style="font-size:2.5rem;margin-bottom:8px;">🧬</div>
                    <div style="font-size:0.82rem;">Result will appear here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Batch Analysis")
    st.markdown('<p style="color:#555;font-size:0.82rem;">Enter one comment per line — up to 200 at once</p>', unsafe_allow_html=True)

    batch_text = st.text_area(
        label="",
        placeholder="This product is great!\nTerrible experience, never again.\nIt was okay, nothing special.\nReally impressed with the quality!",
        height=200,
        key="batch_input",
        label_visibility="collapsed",
    )

    lines = [l.strip() for l in batch_text.splitlines() if l.strip()] if batch_text else []
    st.markdown(
        f'<p style="color:#444;font-size:0.75rem;">{len(lines)} comment{"s" if len(lines)!=1 else ""} detected</p>',
        unsafe_allow_html=True,
    )

    batch_btn = st.button("📦 Analyse All", type="primary")

    if batch_btn:
        if len(lines) == 0:
            st.warning("Please enter at least one comment.")
        elif len(lines) > 200:
            st.error("Maximum 200 comments per batch.")
        else:
            progress_bar = st.progress(0, text="Sending to API...")
            try:
                start = time.perf_counter()
                data  = predict_batch(lines)
                elapsed = round((time.perf_counter() - start) * 1000, 1)
                progress_bar.progress(100, text="Done!")
                time.sleep(0.3)
                progress_bar.empty()

                preds  = data["predictions"]
                counts = Counter(p["label"] for p in preds)

                # Update session stats
                st.session_state.total_analysed += len(preds)
                for p in preds:
                    st.session_state.history.insert(0, {
                        "text"    : p["input_text"][:120],
                        "label"   : p["label"],
                        "time_ms" : p["time_ms"],
                        "ts"      : datetime.now().strftime("%H:%M:%S"),
                    })

                # Summary row
                st.markdown(f"""
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-number" style="color:#fff">{len(preds)}</div>
                        <div class="stat-label">Total</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color:#00C896">{counts.get('Positive',0)}</div>
                        <div class="stat-label">Positive</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color:#F5A623">{counts.get('Neutral',0)}</div>
                        <div class="stat-label">Neutral</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color:#FF4D6D">{counts.get('Negative',0)}</div>
                        <div class="stat-label">Negative</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" style="color:#888">{elapsed}</div>
                        <div class="stat-label">ms total</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Filter controls
                filter_col1, filter_col2 = st.columns([2, 3])
                with filter_col1:
                    filter_label = st.selectbox(
                        "Filter by sentiment",
                        ["All", "Positive", "Neutral", "Negative"],
                        key="batch_filter",
                    )

                filtered = preds if filter_label == "All" else [p for p in preds if p["label"] == filter_label]

                st.markdown(f'<p style="color:#555;font-size:0.78rem;margin-bottom:8px;">Showing {len(filtered)} result{"s" if len(filtered)!=1 else ""}</p>', unsafe_allow_html=True)

                for p in filtered:
                    cfg = LABEL_CONFIG[p["label"]]
                    st.markdown(f"""
                    <div class="batch-item">
                        <span class="batch-label" style="color:{cfg['color']};background:{cfg['bg']};border-color:{cfg['border']}33;">
                            {cfg['emoji']} {p['label']}
                        </span>
                        <span class="batch-text">{p['input_text'][:120]}{'...' if len(p['input_text'])>120 else ''}</span>
                        <span class="meta-tag">{p['time_ms']}ms</span>
                    </div>
                    """, unsafe_allow_html=True)

                # CSV download
                df_export = pd.DataFrame([{
                    "text"    : p["input_text"],
                    "cleaned" : p["cleaned"],
                    "label"   : p["label"],
                    "time_ms" : p["time_ms"],
                } for p in preds])

                st.download_button(
                    label     = "⬇ Download Results as CSV",
                    data      = df_export.to_csv(index=False),
                    file_name = f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime      = "text/csv",
                )

            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                st.error("Lost connection to API.")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Session Analytics")

    hist = st.session_state.history
    if not hist:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:8px;">📊</div>
            <div style="color:#555;font-size:0.88rem;">Run some analyses to see your session stats here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(hist)
        counts = Counter(h["label"] for h in hist)

        # Distribution bar
        pos_pct = counts.get("Positive", 0) / total * 100
        neu_pct = counts.get("Neutral",  0) / total * 100
        neg_pct = counts.get("Negative", 0) / total * 100

        st.markdown("##### Sentiment Distribution")
        st.markdown(f"""
        <div style="margin:1rem 0;">
            <div style="display:flex;border-radius:8px;overflow:hidden;height:32px;gap:2px;">
                <div style="width:{pos_pct:.1f}%;background:#00C896;display:flex;align-items:center;
                            justify-content:center;font-size:0.75rem;font-weight:700;color:#000;
                            min-width:{4 if pos_pct>0 else 0}px;">
                    {f'{pos_pct:.0f}%' if pos_pct > 8 else ''}
                </div>
                <div style="width:{neu_pct:.1f}%;background:#F5A623;display:flex;align-items:center;
                            justify-content:center;font-size:0.75rem;font-weight:700;color:#000;
                            min-width:{4 if neu_pct>0 else 0}px;">
                    {f'{neu_pct:.0f}%' if neu_pct > 8 else ''}
                </div>
                <div style="width:{neg_pct:.1f}%;background:#FF4D6D;display:flex;align-items:center;
                            justify-content:center;font-size:0.75rem;font-weight:700;color:#fff;
                            min-width:{4 if neg_pct>0 else 0}px;">
                    {f'{neg_pct:.0f}%' if neg_pct > 8 else ''}
                </div>
            </div>
            <div style="display:flex;gap:16px;margin-top:8px;font-size:0.78rem;">
                <span style="color:#00C896;">■ Positive {counts.get('Positive',0)}</span>
                <span style="color:#F5A623;">■ Neutral {counts.get('Neutral',0)}</span>
                <span style="color:#FF4D6D;">■ Negative {counts.get('Negative',0)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Avg speed
        avg_ms = sum(h["time_ms"] for h in hist) / total
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-number" style="color:#fff">{total}</div>
                <div class="stat-label">Total analysed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:#00C896">{avg_ms:.1f}</div>
                <div class="stat-label">Avg ms / item</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:#F5A623">{pos_pct:.0f}%</div>
                <div class="stat-label">Positive rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color:#FF4D6D">{neg_pct:.0f}%</div>
                <div class="stat-label">Negative rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recent trend (last 20)
        if len(hist) >= 3:
            st.markdown("##### Recent Trend (last 20)")
            recent   = hist[:20][::-1]
            color_map = {"Positive": "#00C896", "Neutral": "#F5A623", "Negative": "#FF4D6D"}
            dots = "".join([
                f'<span title="{h["label"]}: {h["text"][:40]}..." '
                f'style="display:inline-block;width:18px;height:18px;border-radius:50%;'
                f'background:{color_map[h["label"]]};margin:2px;opacity:0.85;"></span>'
                for h in recent
            ])
            st.markdown(f'<div style="margin:0.5rem 0;">{dots}</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#444;font-size:0.72rem;">Each dot = one prediction (oldest → newest, left to right)</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — History
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Prediction History")

    hist = st.session_state.history
    if not hist:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:8px;">🕐</div>
            <div style="color:#555;font-size:0.88rem;">Your prediction history will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Filter
        hist_filter = st.selectbox(
            "Filter",
            ["All", "Positive", "Neutral", "Negative"],
            key="hist_filter",
            label_visibility="collapsed",
        )
        filtered_hist = hist if hist_filter == "All" else [h for h in hist if h["label"] == hist_filter]

        st.markdown(f'<p style="color:#555;font-size:0.78rem;margin-bottom:10px;">{len(filtered_hist)} record{"s" if len(filtered_hist)!=1 else ""}</p>', unsafe_allow_html=True)

        for h in filtered_hist[:50]:   # cap at 50 for performance
            cfg = LABEL_CONFIG[h["label"]]
            st.markdown(f"""
            <div class="batch-item">
                <span class="meta-tag" style="color:#444;">{h['ts']}</span>
                <span class="batch-label" style="color:{cfg['color']};background:{cfg['bg']};border-color:{cfg['border']}33;">
                    {cfg['emoji']} {h['label']}
                </span>
                <span class="batch-text">{h['text']}</span>
                <span class="meta-tag">{h['time_ms']}ms</span>
            </div>
            """, unsafe_allow_html=True)

        # Export history
        df_hist = pd.DataFrame(filtered_hist)
        st.download_button(
            label     = "⬇ Export History as CSV",
            data      = df_hist.to_csv(index=False),
            file_name = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime      = "text/csv",
        )