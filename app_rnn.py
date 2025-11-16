import streamlit as st
import re
import tempfile
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# ----------------------
# Configuration defaults
# ----------------------
DEFAULT_NUM_WORDS = 10000
DEFAULT_MAX_LEN = 200
MODEL_FILENAME = "imdb_sentiment_analysis.keras"  # fallback filename to look for in working dir

# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def get_word_index():
    # note: Keras IMDB word index maps word->integer (starting at 1)
    return imdb.get_word_index()

def preprocess_text(text, word_index, num_words=DEFAULT_NUM_WORDS, max_len=DEFAULT_MAX_LEN):
    # Basic cleaning
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()

    seq = []
    for w in words:
        idx = word_index.get(w)
        if idx is None:
            seq.append(2)  # unknown token used by Keras IMDB
        else:
            # Keras reserves indices 0,1,2 for padding/start/unknown in built-in dataset
            # add 3 offset as in Keras examples
            idx = idx + 3
            if idx < num_words:
                seq.append(idx)
            else:
                seq.append(2)

    padded = pad_sequences([seq], maxlen=max_len)
    return padded

# ----------------------
# Model loader
# ----------------------
@st.cache_resource
def load_rnn_model_from_file(path):
    return load_model(path)

# ----------------------
# Streamlit App UI
# ----------------------
st.set_page_config(page_title="IMDB RNN Sentiment", layout="centered")
st.title("IMDB RNN Sentiment Predictor")
st.write("Enter a movie review (or paste multiple lines). The app will preprocess the text using the Keras IMDB word index and predict sentiment using your RNN model.")

# Sidebar: Model Upload Only
uploaded_model = st.sidebar.file_uploader("Upload your trained model (.keras)", type=["keras"], help="If you don't upload, the app will try to load 'imdb_sentiment_analysis.keras' from the working directory.")

# Fixed preprocessing parameters
num_words = DEFAULT_NUM_WORDS
max_len = DEFAULT_MAX_LEN

model = None
if uploaded_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.getbuffer())
        tmp.flush()
        st.sidebar.success("Model uploaded — loading...")
        try:
            model = load_rnn_model_from_file(tmp.name)
            st.sidebar.success("Model loaded from uploaded file")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

if model is None:
    try:
        model = load_rnn_model_from_file(MODEL_FILENAME)
        st.sidebar.success(f"Loaded model from working dir: {MODEL_FILENAME}")
    except Exception:
        st.sidebar.info("No model loaded yet. Upload a .h5 model or place 'imdb_sentiment_analysis.keras' in working directory.")

# Load IMDB word index (cached)
word_index = get_word_index()

# Input area
st.subheader("Enter review text")
review_text = st.text_area("Review", value="The movie was absolutely wonderful and touching.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict"):
        if model is None:
            st.error("No model available. Upload a model (.h5) or place 'imdb_sentiment_analysis.keras' in the working directory.")
        elif not review_text.strip():
            st.warning("Please enter some text to predict on.")
        else:
            try:
                seq = preprocess_text(review_text, word_index, num_words=num_words, max_len=max_len)
                pred = model.predict(seq)[0][0]
                label = "Positive" if pred >= 0.5 else "Negative"
                st.metric(label=f"Sentiment: {label}", value=f"{pred:.4f}")
                st.write(f"**Label:** {label} — Probability: {pred:.4f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
with col2:
    st.info("Tips:\n - Make sure the model was trained with the same `num_words` (vocab size) and `max_len` used here.\n - Keras IMDB reserves indices 0,1,2 for padding/start/unknown.\n - If predictions are garbage, try using the original preprocessing and same hyperparameters used during training.")

# Optional: sample reviews to test quickly
st.subheader("Try sample reviews")
if st.button("Run samples"):
    samples = [
        "An excellent movie with wonderful performances and a moving story.",
        "Terrible. I wasted my time. The plot was dumb and acting was awful.",
        "It was okay — some good parts, some boring scenes. Not the best, not the worst."
    ]
    if model is None:
        st.error("No model loaded — can't run samples.")
    else:
        for s in samples:
            seq = preprocess_text(s, word_index, num_words=num_words, max_len=max_len)
            p = model.predict(seq)[0][0]
            st.write(f"**Review:** {s}\n**Score:** {p:.4f} — {'Positive' if p>=0.5 else 'Negative'}\n---")

# Footer: how to run
st.write("---")
st.write("**How to run locally**:\n1. Save this file as `streamlit_imdb_rnn.py`.\n2. Install dependencies: `pip install streamlit tensorflow` (use a Python environment).\n3. Put your trained model as `imdb_sentiment_analysis.keras` in the same folder or upload it in the sidebar.\n4. Run: `streamlit run streamlit_imdb_rnn.py`.")

st.write("**Deployment**: you can deploy on Streamlit Community Cloud (share a GitHub repo with this file + the model or model loader). Alternatively use any container/VM and run the same `streamlit run` command.")
