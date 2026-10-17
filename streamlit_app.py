import streamlit as st
import pandas as pd
import sys
import os

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath("."))
from src.predict import get_predictor

# Page Configuration
st.set_page_config(
    page_title="Shakespeare Next-Word Prediction",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme Styles
LIGHT_THEME_CSS = """
<style>
/* Light Theme Global Overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.85) !important;
    backdrop-filter: blur(8px);
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0;
}

/* Custom Header Card */
.main-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.05);
}

.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.sub-title {
    color: #64748B;
    font-size: 1.02rem;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Stat Pills */
.stat-pill {
    display: inline-flex;
    align-items: center;
    background: #EEF2FF;
    color: #4F46E5;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid #E0E7FF;
    margin-right: 8px;
    margin-top: 10px;
}

/* Custom Card Container */
.content-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04);
}

/* Prediction Word Pill */
.pred-badge {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.pred-badge:hover {
    background-color: #EEF2FF;
    border-color: #818CF8;
}

.token-pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 3px;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
}

.token-known {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

.token-oov {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}

.token-pad {
    background-color: #F1F5F9;
    color: #64748B;
    border: 1px solid #E2E8F0;
}

/* Primary Button Styling */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.15s ease-in-out;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0px 0px;
    padding: 10px 18px;
    font-weight: 600;
}
</style>
"""

st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# Load Model Predictor
@st.cache_resource(show_spinner=False)
def load_model():
    return get_predictor()

with st.spinner("Loading Shakespeare LSTM Model & Vocabulary..."):
    predictor = load_model()

# Header Section
st.markdown(
    f"""
    <div class="main-header">
        <h1 class="main-title">✒️ Shakespeare Next-Word Predictor</h1>
        <p class="sub-title">Interactive 2-Layer LSTM neural language model trained on the Tiny Shakespeare corpus.</p>
        <div>
            <span class="stat-pill">📖 Vocab: {predictor.vocab_len:,} words</span>
            <span class="stat-pill">🧠 2-Layer LSTM (Hidden: 256)</span>
            <span class="stat-pill">🪟 Window: 10 Words</span>
            <span class="stat-pill">🎯 Perplexity: 320.1</span>
            <span class="stat-pill">⚡ Device: {predictor.device.type.upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = "to be or not to be that is the"

def clear_prompt_callback():
    st.session_state["prompt_input"] = ""

def set_preset_callback(preset_text):
    st.session_state["prompt_input"] = preset_text

def append_word_callback(word):
    curr = st.session_state.get("prompt_input", "").strip()
    st.session_state["prompt_input"] = f"{curr} {word}".strip()

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Inference Controls")
    
    top_k = st.slider("Top-K Candidates", min_value=3, max_value=15, value=5, step=1,
                      help="Number of most probable next words to predict.")
    
    temperature = st.slider("Sampling Temperature", min_value=0.1, max_value=1.8, value=1.0, step=0.05,
                            help="Lower values make predictions more confident and focused; higher values increase diversity.")
    
    filter_unk = st.checkbox("Filter `<UNK>` Tokens", value=True,
                             help="Suppress out-of-vocabulary (<UNK>) predictions for more coherent Shakespearean phrasing.")

    st.markdown("---")
    st.markdown("### 🎭 Famous Shakespeare Presets")
    st.caption("Click any quote to test the model:")

    presets = [
        "to be or not to be that is the",
        "shall i compare thee to a",
        "all the world is a",
        "friends romans countrymen lend me your",
        "now is the winter of our",
        "some are born great some achieve"
    ]

    for p in presets:
        st.button(f"📜 {p[:30]}...", key=f"preset_{p}", on_click=set_preset_callback, args=(p,), use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About This Model")
    st.markdown(
        """
        - **Corpus:** 1.1M characters (Tiny Shakespeare)
        - **Embedding:** 128 dimensions
        - **LSTM:** 2 stacked layers with 256 units
        - **Dropout:** 0.4
        - **Context Window:** 10 words (sliding)
        - **Next-Word Accuracy:** 12.79%
        """
    )

# Main Application Tabs
tab_predict, tab_tokens, tab_vocab, tab_arch = st.tabs([
    "🔮 Predict & Generate",
    "🪟 Context Window (10 Tokens)",
    "📚 Vocabulary Explorer",
    "📊 Model Architecture & Stats"
])

# ----------------- TAB 1: PREDICT & GENERATE -----------------
with tab_predict:
    col_input, col_pred = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### 💬 Your Text Prompt")
        user_prompt = st.text_area(
            "Enter sequence of words:",
            key="prompt_input",
            height=120,
            help="Type your text. The model analyzes the last 10 words to predict the next word."
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.button("🧹 Clear Prompt", on_click=clear_prompt_callback, use_container_width=True)
        with c2:
            if st.button("🔄 Refresh Prediction", use_container_width=True):
                st.rerun()

        st.markdown("---")
        st.markdown("#### ⚡ Continuous Text Generation")
        st.caption("Auto-generate a sequence of consecutive words starting from your prompt.")
        
        gen_words_count = st.slider("Words to Generate", min_value=5, max_value=40, value=15, step=5)
        gen_strategy = st.radio(
            "Generation Strategy:",
            options=["top_k", "temperature", "greedy"],
            format_func=lambda x: {
                "top_k": "Top-K Sampling (Balanced & Natural)",
                "temperature": "Temperature Sampling (Creative & Varied)",
                "greedy": "Greedy Argmax (Deterministic)"
            }[x],
            horizontal=False
        )

        if st.button("✨ Generate Shakespearean Passage", type="primary", use_container_width=True):
            if not st.session_state["prompt_input"].strip():
                st.warning("Please enter a prompt first.")
            else:
                with st.spinner("Generating sequence..."):
                    gen_result = predictor.generate_sequence(
                        seed_text=st.session_state["prompt_input"],
                        num_words=gen_words_count,
                        temperature=temperature,
                        filter_unk=filter_unk,
                        strategy=gen_strategy
                    )
                st.success("Generation Complete!")
                st.markdown(
                    f"""
                    <div style="background-color: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #4F46E5; border-radius: 8px; padding: 18px; margin-top: 12px; font-family: Georgia, serif; font-size: 1.12rem; line-height: 1.6; color: #1E293B;">
                        <span style="color: #64748B;">{gen_result['seed']}</span> 
                        <span style="color: #4F46E5; font-weight: 600;">{' '.join(gen_result['generated_words'])}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col_pred:
        st.markdown("#### 🎯 Next Word Candidates")
        st.caption("Click any word below to append it to your prompt and continue predicting!")

        current_prompt = st.session_state.get("prompt_input", "").strip()
        if current_prompt:
            res = predictor.predict_next_word(
                prompt=current_prompt,
                top_k=top_k,
                filter_unk=filter_unk,
                temperature=temperature
            )

            preds = res["predictions"]

            if preds:
                for idx, item in enumerate(preds):
                    word = item["word"]
                    percent = item["percent"]
                    w_idx = item["index"]

                    # Display clickable button for each prediction
                    btn_label = f"➕  {word}  —  {percent}%"
                    st.button(
                        btn_label,
                        key=f"append_{word}_{idx}",
                        on_click=append_word_callback,
                        args=(word,),
                        use_container_width=True
                    )

                # Probability Distribution Chart
                st.markdown("##### 📊 Probability Breakdown")
                df_preds = pd.DataFrame({
                    "Word": [p["word"] for p in preds],
                    "Probability (%)": [p["percent"] for p in preds]
                }).set_index("Word")

                st.bar_chart(df_preds, height=220, color="#4F46E5")
            else:
                st.info("No candidates available.")
        else:
            st.info("💡 Enter a prompt on the left or select a preset to see predictions.")

# ----------------- TAB 2: SLIDING CONTEXT WINDOW -----------------
with tab_tokens:
    st.markdown("#### 🪟 10-Word Sliding Context Window Visualizer")
    st.markdown(
        """
        The LSTM model expects an input vector of exactly **10 tokens**. 
        - If the prompt has fewer than 10 words, it is left-padded with `<PAD>`.
        - If the prompt has more than 10 words, only the **last 10 words** are fed into the network.
        """
    )

    if st.session_state.prompt_input.strip():
        _, _, debug_tokens = predictor.tokenize_and_encode(st.session_state.prompt_input)

        st.markdown("##### Current Window Tokens:")
        html_tokens = '<div style="margin-bottom: 16px;">'
        for dt in debug_tokens:
            w = dt["word"]
            status = dt["status"]
            cls_name = f"token-{status}"
            badge_desc = "Known" if status == "known" else ("OOV (<UNK>)" if status == "oov" else "Padding")
            html_tokens += f'<span class="token-pill {cls_name}" title="{badge_desc} (ID: {dt["index"]})"><strong>{w}</strong> <small style="opacity:0.75;">#{dt["index"]}</small></span>'
        html_tokens += '</div>'
        st.markdown(html_tokens, unsafe_allow_html=True)

        df_tokens = pd.DataFrame(debug_tokens)
        df_tokens.columns = ["Token String", "Vocabulary ID", "Token Status"]
        st.dataframe(df_tokens, use_container_width=True, hide_index=True)
    else:
        st.info("Enter text in the Predict tab to inspect tokens.")

# ----------------- TAB 3: VOCABULARY EXPLORER -----------------
with tab_vocab:
    st.markdown("#### 📚 Shakespeare Corpus Vocabulary Explorer")
    st.markdown(
        f"The training vocabulary contains **{predictor.vocab_len:,}** unique words "
        "extracted from *Tiny Shakespeare* (words occurring at least twice)."
    )

    col_vsearch, col_vtop = st.columns([1, 1], gap="medium")

    with col_vsearch:
        st.markdown("##### 🔍 Search Lexicon")
        search_query = st.text_input("Find word in vocabulary:", placeholder="e.g. king, queen, honor, death")
        if search_query:
            results = predictor.search_vocab(search_query, limit=25)
            if results:
                st.success(f"Found {len(results)} matches:")
                df_res = pd.DataFrame(results)
                df_res.columns = ["Word", "Token Index"]
                st.dataframe(df_res, use_container_width=True, hide_index=True)
            else:
                st.warning(f"'{search_query}' not found in vocabulary. (Would be encoded as <UNK>)")

    with col_vtop:
        st.markdown("##### 🏆 Most Frequent Words in Shakespeare")
        if predictor.top_words:
            top_df = pd.DataFrame(predictor.top_words[:20], columns=["Word", "Corpus Count"]).set_index("Word")
            st.bar_chart(top_df, height=320, color="#6366F1")

# ----------------- TAB 4: ARCHITECTURE & METRICS -----------------
with tab_arch:
    st.markdown("#### 📊 Model Architecture & Training Results")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Vocabulary Size", f"{predictor.vocab_len:,}")
    col_m2.metric("Perplexity", "320.1")
    col_m3.metric("Top-1 Accuracy", "12.79%")
    col_m4.metric("Best Val Loss", "5.77")

    st.markdown("---")
    st.markdown("##### 🧱 Neural Network Layer Diagram")
    
    st.markdown(
        """
        ```
        [ Input: Sequence of 10 Word Indices ]
                          │
                          ▼
        [ Embedding Layer: 10,113 Vocab → 128 Dims ]
                          │
                          ▼
        [ LSTM Layer 1: 128 → 256 Units, Dropout = 0.4 ]
                          │
                          ▼
        [ LSTM Layer 2: 256 → 256 Units, Dropout = 0.4 ]
                          │
                          ▼
        [ Take Last Timestep: Shape (batch, 256) ]
                          │
                          ▼
        [ Linear Projection Head: 256 → 10,113 Vocab Logits ]
                          │
                          ▼
        [ Softmax → Next Word Probability Distribution ]
        ```
        """
    )
