

# 📘 **Abstractive Text Summarization using T5 / BART**

This project implements **abstractive text summarization** using powerful **Encoder–Decoder architectures (T5 and BART)**.
The model is **fine-tuned on the CNN/DailyMail news dataset** to generate **concise, human-like summaries** of long articles.

---

## ✨ **Features**

✔ **Preprocessing of article–summary pairs**
✔ **Fine-tuning T5 or BART** using Hugging Face Transformers
✔ **Evaluation using ROUGE-1, ROUGE-2, ROUGE-L**
✔ **Qualitative comparison** of predicted vs. reference summaries
✔ **Streamlit demo** for real-time summarization *(optional)*

---

## 📂 **Dataset**

**CNN/DailyMail dataset:**
[https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

The dataset contains:

* **Article:** Full news story
* **Highlights:** Human-written summary

**Task:** Generate a **short abstractive summary** from a long article.

---

## 🛠 **Model Architecture**

This project uses an **Encoder–Decoder Transformer**:

### 🔹 **T5 (Text-to-Text Transfer Transformer)**

* Unified text-to-text format
* Strong abstractive summarization performance

### 🔹 **BART (Bidirectional + Autoregressive Transformer)**

* Robust denoising autoencoder
* Excellent for long-document summarization

---

## 🚀 **Training Pipeline**

### **1. Preprocessing**

* Load dataset
* Clean text *(HTML, whitespace, special characters)*
* Map **article → summary** pairs
* Tokenize using model tokenizer
* Create PyTorch datasets

### **2. Fine-Tuning**

* HuggingFace **Trainer API**
* **Loss:** Cross-entropy
* **Batch size:** 2–4
* **Epochs:** 2–3
* **Learning rate:** 3e-5

### **3. Evaluation**

Metrics computed:

* **ROUGE-1**
* **ROUGE-2**
* **ROUGE-L**

Outputs stored in the `results/` directory.


---

## 🧪 **Usage (Inference)**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"  # or "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained("your_finetuned_model_path")

text = "Your article text here..."
inputs = tokenizer("summarize: " + text, return_tensors="pt",
                   max_length=1024, truncation=True)

summary_ids = model.generate(inputs["input_ids"],
                             max_length=150, min_length=40)

print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

---

## 🖥️ **Optional: Streamlit App**

**File:** `app.py`

```
streamlit run app.py
```

This launches a **simple web interface** where users can paste text and get instant summaries.

---

## 📦 **Install Requirements**

```
pip install -r requirements.txt
```

---

## 📁 **Project Structure**

```
├── summarizer.ipynb
├── app.py
├── requirements.txt
├── README.md
└── results/
```

---

## 📜 **License**

**MIT License**

