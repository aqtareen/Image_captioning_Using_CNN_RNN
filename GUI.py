import keras
keras.config.enable_unsafe_deserialization()
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding, LSTM,
    Concatenate, Activation, Multiply
)
import tensorflow as tf


def build_model(vocab_size, max_len):
    img_input  = Input(shape=(49, 512), name="image_input")
    img_dense  = Dense(512, activation='relu')(img_input)
    img_drop   = Dropout(0.3)(img_dense)
    seq_input  = Input(shape=(max_len,), name="seq_input")
    seq_embed  = Embedding(vocab_size, 256, mask_zero=True)(seq_input)
    seq_drop   = Dropout(0.3)(seq_embed)
    seq_lstm   = LSTM(512, use_cudnn=False)(seq_drop)
    seq_expanded = tf.keras.layers.RepeatVector(49)(seq_lstm)
    combined   = Concatenate()([img_drop, seq_expanded])
    score      = Dense(256, activation='tanh')(combined)
    energy     = Dense(1)(score)
    weights    = tf.keras.layers.Softmax(axis=1)(energy)
    context    = Multiply()([img_drop, weights])
    context    = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        output_shape=(512,)
    )(context)
    merged     = Concatenate()([context, seq_lstm])
    decoder1   = Dense(512, activation='relu')(merged)
    decoder2   = Dropout(0.3)(decoder1)
    output     = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[img_input, seq_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Image Captioner", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0a;
    color: #e8e8e8;
    font-family: 'Syne', sans-serif;
}
.stApp { background-color: #0a0a0a; }
h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 0;
}
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.caption-box {
    background: #111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #c8ff00;
    border-radius: 8px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #e8e8e8;
    line-height: 1.7;
}
.caption-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #c8ff00;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.stButton > button {
    background: #c8ff00;
    color: #0a0a0a;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #d4ff33;
    transform: translateY(-1px);
}
.stSelectbox label, .stSlider label {
    color: #a0a0a0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
.method-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
#MainMenu, footer, header {visibility: hidden;}
.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Caption methods ────────────────────────────────────────
def generate_caption_greedy(model, feature, word_to_idx, idx_to_word, max_len):
    caption = "startseq"
    for _ in range(max_len + 20):
        seq      = [word_to_idx.get(w, 0) for w in caption.split()]
        seq      = pad_sequences([seq], maxlen=max_len)
        pred     = model.predict([feature, seq], verbose=0)
        next_idx = np.argmax(pred)
        next_word = idx_to_word.get(next_idx, "")
        if next_word == "endseq" or not next_word:
            break
        caption += " " + next_word
    return caption.replace("startseq", "").strip()


def generate_caption_beam(model, feature, word_to_idx, idx_to_word, max_len, beam_width=3):
    sequences = [[["startseq"], 0.0]]
    for _ in range(max_len + 20):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == "endseq":
                all_candidates.append([seq, score])
                continue
            encoded = pad_sequences(
                [[word_to_idx.get(w, 0) for w in seq]], maxlen=max_len
            )
            pred  = model.predict([feature, encoded], verbose=0)[0]
            top_k = np.argsort(pred)[-beam_width:]
            for idx in top_k:
                word = idx_to_word.get(idx, "")
                candidate = [seq + [word], score - np.log(pred[idx] + 1e-10)]
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
    best = sequences[0][0]
    return " ".join([w for w in best if w not in ("startseq", "endseq", "")])


def generate_caption_temperature(model, feature, word_to_idx, idx_to_word, max_len, temperature=0.7):
    caption = [word_to_idx.get('startseq', 0)]
    for _ in range(max_len + 20):
        padded   = pad_sequences([caption], maxlen=max_len)
        preds    = model.predict([feature, padded], verbose=0)[0]
        preds    = np.log(preds + 1e-10) / temperature
        preds    = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(len(preds), p=preds)
        next_word = idx_to_word.get(next_idx, '')
        if next_word == 'endseq' or not next_word:
            break
        caption.append(next_idx)
    return ' '.join([idx_to_word.get(i, '') for i in caption[1:]]).strip()


# ── Load resources ─────────────────────────────────────────
@st.cache_resource
def load_caption_model():
    with open("Output/Model/word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    with open("Output/Model/config.pkl", "rb") as f:
        config     = pickle.load(f)
        vocab_size = config['vocab_size']
        max_len    = config['max_len']
    model = build_model(vocab_size, max_len)
    model.load_weights("Output/Model/model_weights.weights.h5")
    return model

@st.cache_resource
def load_feature_extractor():
    base      = VGG16(weights='imagenet')
    extractor = Model(inputs=base.input, outputs=base.get_layer('block5_pool').output)
    return extractor

@st.cache_resource
def load_vocab():
    with open("Output/Model/word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    with open("Output/Model/idx_to_word.pkl", "rb") as f:
        idx_to_word = pickle.load(f)
    return word_to_idx, idx_to_word

def extract_feature(pil_image, extractor):
    img  = pil_image.resize((224, 224)).convert("RGB")
    img  = img_to_array(img)
    img  = preprocess_input(img)
    img  = np.expand_dims(img, axis=0)
    feat = extractor.predict(img, verbose=0)
    return feat.reshape(1, 49, 512)


# ── UI ─────────────────────────────────────────────────────
st.markdown("<h1>Caption Generator Using CNN and RNN</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">22i-2127 · Abdul Qadir Tareen · DLP BCS 8A</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Decoding method picker ─────────────────────────────────
st.markdown('<p class="caption-label">Decoding Method</p>', unsafe_allow_html=True)
method = st.selectbox(
    "Decoding Method",
    ["Greedy Search", "Beam Search", "Temperature Sampling"],
    label_visibility="collapsed"
)

# ── Method-specific controls ───────────────────────────────
beam_width  = 3
temperature = 0.7

if method == "Beam Search":
    st.markdown('<p class="caption-label">Beam Width</p>', unsafe_allow_html=True)
    beam_width = st.slider("Beam Width", min_value=1, max_value=10,
                           value=3, step=1, label_visibility="collapsed")
    st.caption(f"Higher beam width = more candidates considered = better but slower")

elif method == "Temperature Sampling":
    st.markdown('<p class="caption-label">Temperature</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", min_value=0.5, max_value=1.5,
                            value=0.7, step=0.1, label_visibility="collapsed")
    st.caption("0.5 = conservative  ·  1.0 = balanced  ·  1.5 = creative")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, use_column_width=True)
    st.markdown("")

    if st.button("Generate Caption"):
        with st.spinner("Analysing image..."):
            try:
                model                    = load_caption_model()
                extractor                = load_feature_extractor()
                word_to_idx, idx_to_word = load_vocab()
                max_len                  = model.input_shape[1][1]
                feature                  = extract_feature(pil_img, extractor)

                if method == "Greedy Search":
                    caption = generate_caption_greedy(
                        model, feature, word_to_idx, idx_to_word, max_len
                    )
                elif method == "Beam Search":
                    caption = generate_caption_beam(
                        model, feature, word_to_idx, idx_to_word, max_len, beam_width
                    )
                else:
                    caption = generate_caption_temperature(
                        model, feature, word_to_idx, idx_to_word, max_len, temperature
                    )

                method_tag = (f"Beam Search · width={beam_width}" if method == "Beam Search"
                              else f"Temperature · t={temperature}" if method == "Temperature Sampling"
                              else "Greedy Search")

                st.markdown(f'<p class="caption-label">Generated Caption — {method_tag}</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="caption-box">{caption}</div>',
                            unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(f"Missing file: {e}")