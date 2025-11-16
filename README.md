# 🎬 IMDB Sentiment Analysis using Simple RNN

This project demonstrates a **Simple Recurrent Neural Network (RNN)** model built using **TensorFlow/Keras** to classify movie reviews from the IMDB dataset as **Positive** or **Negative**.  
The trained model achieves **around 95% accuracy** on the test set.

The repository includes:
- 📓 `RNN.ipynb` – Jupyter Notebook used for training  
- 🤖 `imdb_sentiment_analysis.keras` – saved trained RNN model  
- 🌐 `app_rnn.py` – Streamlit web app for real-time prediction  
- 📂 Clean project structure for easy deployment

---

## 🚀 Features

### 🔹 Model Training
- Uses **TensorFlow/Keras IMDB dataset**
- Text preprocessing with:
  - Keras IMDB word index
  - Custom cleaning + token-to-index mapping
  - Sequence padding to fixed length (`max_len = 200`)
- Simple yet powerful architecture:
  - **Embedding Layer**
  - **SimpleRNN Layer (128 units)**
  - **Dense Output Layer (Sigmoid)**  
- Achieves **~95% accuracy** after training

---

## 🌐 Streamlit App

The Streamlit app (`app_rnn.py`) allows users to:

- Input custom movie reviews  
- Automatically preprocess the text  
- Load the **.keras** model  
- Predict sentiment in real time  
- Display probability + label  

## 📊 Dataset Details

The model uses the **IMDB Movie Review Dataset**:
- 50,000 labeled reviews  
- 25,000 training / 25,000 testing  
- Balanced: 50% positive, 50% negative  
- Pre-tokenized into integer sequences  
- `num_words = 10000` vocabulary size  

---

## 📈 Model Accuracy

Final test accuracy:  
**✔ ~95%** using SimpleRNN  
Hyperparameters:
- Embedding dimension: 64  
- RNN units: 128  
- Max sequence length: 200  
- Batch size: 64  
- 5 epochs  

---
