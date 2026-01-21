# 🧠 HumanIntent-AI

A deep learning-based project focused on detecting human intention and violence using sequential video data.

---

## 📌 Description
HumanIntent-AI uses a **CNN-LSTM architecture** to analyze video frames and classify human actions as **violent or non-violent**.  
The CNN extracts spatial features from individual frames, while the LSTM captures **temporal dependencies** across frame sequences.

The main application logic and model implementation are organized inside the **`src/` directory** for clarity and modularity.

---

## 📂 Dataset Used
- **AIRT Lab – Automatic Violence Detection in Videos Dataset**
- Contains labeled video clips of **violent and non-violent human activities**
- Used for training and evaluation of the CNN-LSTM model

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/airtlab/automatic-violence-detection-in-videos

> ⚠️ *The dataset is not included in this repository due to size and licensing constraints.*

---

## ✨ Features
- Human intention and violence detection
- Sequence-based video analysis
- CNN + LSTM deep learning architecture
- Real-time prediction interface using Streamlit
- Modular and organized code structure

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- NumPy  
- Pandas  
- Streamlit  
- Matplotlib  

---

## 📊 Results & Observations

### 🔹 Training Performance
- **Training Loss:** Consistently decreases across epochs
- **Training Accuracy:** Reaches approximately **99%**
- **Validation Accuracy:** Stabilizes around **90–95%**

These results show that the model effectively learns **spatial–temporal patterns** in video data, with minor fluctuations due to dataset complexity.

### 🔹 Training Graphs
The following graphs were generated during training:
- Training Loss vs Epochs
- Training Accuracy vs Validation Accuracy

📌 *(Add the training metrics image here)*  
```md
![Training Metrics](assest/graph.png)
