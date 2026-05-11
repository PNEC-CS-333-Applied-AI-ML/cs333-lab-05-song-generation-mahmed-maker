# Lab 05 – Music Generation with RNNs
**MIT Introduction to Deep Learning**
---
## Run the codes to see the outputs, with outputs the file becomes too large it is not uploading.
## Overview

This lab builds a character-level Recurrent Neural Network (RNN) that learns patterns from Irish folk songs written in [ABC notation](https://en.wikipedia.org/wiki/ABC_notation) and then generates brand new music from scratch. The model is trained to predict the next character in a sequence, and after enough training it produces valid, playable ABC music.

---

## How It Works

1. **Dataset** — Thousands of Irish folk songs in ABC notation are loaded and joined into one long string.
2. **Vectorization** — Every unique character is mapped to an integer index, turning the text into a numeric array.
3. **Batching** — Random input/output sequence pairs are created; the output is the input shifted one character to the right (next-character prediction).
4. **Model** — A 3-layer Sequential model is built using Keras:
   - `Embedding` layer — maps character indices to dense vectors
   - `LSTM` layer — learns temporal/sequential patterns
   - `Dense` layer — outputs a probability over the full vocabulary
5. **Training** — The model is trained using `sparse_categorical_crossentropy` loss and the Adam optimizer. Weights are saved as checkpoints.
6. **Generation** — The trained model is rebuilt with `batch_size=1`, weights are restored, and new ABC text is generated character-by-character starting from a seed string (`"X"`).
7. **Playback** — Generated ABC text is converted to MIDI and then to WAV using `abcmidi` and `timidity`.

---

## Model Architecture

| Layer | Type | Details |
|---|---|---|
| 1 | Embedding | `vocab_size` → `embedding_dim=256` |
| 2 | LSTM | `rnn_units=1024`, stateful, returns sequences |
| 3 | Dense | `vocab_size` outputs (raw logits) |

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Training iterations | 3000 |
| Batch size | 8 |
| Sequence length | 100 |
| Learning rate | 5e-3 |
| Embedding dim | 256 |
| RNN units | 1024 |

---

## Requirements

- Python 3.x
- TensorFlow 2.x (GPU recommended)
- `mitdeeplearning` package
- `numpy`, `scipy`, `tqdm`
- System tools: `abcmidi`, `timidity`

Install dependencies:
```bash
pip install mitdeeplearning tensorflow numpy scipy tqdm
apt-get install abcmidi timidity
```

---

## How to Run

1. Open the notebook in **Google Colab** (recommended — requires GPU).
2. Go to `Runtime > Change Runtime Type > GPU`.
3. Run all cells top to bottom.
4. After training completes, the model will generate a song and save it as `output_0.wav`.

---

## Files

| File | Description |
|---|---|
| Main notebook with all code and answers |
| `output_0.wav` | Generated audio file (created at runtime) |
| `training_checkpoints/` | Saved model weights (created at runtime) |

---

## Results

After training, the model generates ABC-formatted music that follows valid structural patterns — correct headers (`X:`, `T:`, `K:`), proper note syntax, and bar lines. The longer the training, the more coherent and musical the output becomes.

---
