# Image Captioning Using CNN + RNN

This project combines computer vision and natural language processing to automatically generate captions for images. Image features are extracted using a VGG16 CNN with spatial Bahdanau attention, and captions are generated using an LSTM-based RNN decoder.

---

## Architecture

- **Feature Extractor:** VGG16 (`block5_pool` layer) → 49 spatial regions of 512-dim each
- **Attention:** Bahdanau spatial attention over 49 image regions
- **Decoder:** LSTM (512 units) + Dense decoder
- **Dataset:** Flickr8k
- **Training:** 20 epochs, Adam optimizer, categorical crossentropy

---

## Project Structure

```
Image_captioning_Using_CNN_RNN/
│
├── Output/
│   └── Model/
│       ├── model_weights.weights.h5   # Trained model weights
│       ├── word_to_idx.pkl            # Word → index vocabulary
│       ├── idx_to_word.pkl            # Index → word vocabulary
│       └── config.pkl                 # vocab_size and max_len
│
├── GUI.py                             # Streamlit web app
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
streamlit run GUI.py
```

---

## Known Issues & Fixes

### 1. `ValueError: bad marshal data (unknown type code)`
**Cause:** The model was saved on Kaggle (Linux, Python 3.12.12) and loaded locally on a different Python patch version. Keras serializes `Lambda` layer bytecode using `marshal`, which is Python version-specific.

**Fix:** Save only the weights on Kaggle and rebuild the model architecture locally:
```python
# On Kaggle
model.save_weights("/kaggle/working/model_weights.weights.h5")
```
```python
# In GUI.py
model = build_model(vocab_size, max_len)
model.load_weights("Output/Model/model_weights.weights.h5")
```

---

### 2. `NameError: name 'tf' is not defined` (Lambda layer)
**Cause:** When loading `best_model.keras` directly, the deserialized Lambda function loses its reference to `tf` since it was defined in a different Python process.

**Fix:** Same as above — use `build_model` + `load_weights` instead of `load_model`. Since `tf` is imported at the top of `GUI.py`, the Lambda defined inside `build_model` will have access to it.

---

### 3. `NameError: name 'vocab_size' is not defined`
**Cause:** Leftover lines from the Kaggle notebook were copied into `GUI.py`:
```python
model = build_model(vocab_size, max_len)  # ← not valid at module level in GUI
model.summary()
plot_model(model, show_shapes=True)
```
**Fix:** Delete those three lines from `GUI.py`. The model is built inside `load_caption_model()` where `vocab_size` and `max_len` are loaded from `config.pkl`.

---

### 4. `from logging import config` conflict
**Cause:** This import shadows the local `config` variable used to load `config.pkl`.

**Fix:** Remove `from logging import config` from the top of `GUI.py`.

---

### 5. Feature extractor slow on first run
**Cause:** VGG16 weights (~550MB) are loaded from disk on cold start.

**Fix:** This is expected behaviour. Thanks to `@st.cache_resource`, VGG16 only loads once per session. Subsequent caption generations are fast.

---

### 6. `ValueError: Input shape (1, 1, 49, 512)` instead of `(1, 49, 512)`
**Cause:** Features stored in `features_spatial.pkl` have an extra batch dimension from extraction.

**Fix:** Index with `[0]` when retrieving:
```python
feature = features_spatial[img_name][0]  # (1, 1, 49, 512) → (1, 49, 512)
```