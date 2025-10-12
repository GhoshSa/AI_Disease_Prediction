# 🧠 Symptom-to-Disease Prediction with Interactive Conversation

This project implements a **Neural Network (MLP)** from scratch using **NumPy** to predict diseases from symptoms.  
It integrates an **interactive chatbot-like loop** that asks users for symptoms and do its prediction.  


## ✨ Features
- Custom **MLP** with:
  - ReLU activations  
  - Dropout regularization  
  - Softmax output  
  - Cross-entropy loss  
- **Adam optimizer** with learning rate decay for stable training  
- **Symptom-to-disease dataset support** (`Training.csv`, `Testing.csv`)  
- **Interactive diagnosis loop**:
  - User types symptoms (`itching, skin rash`)  
  - Model suggests possible diseases  
  - If confidence is low, it asks for more symptoms until confident  


## 📂 Dataset
⚠️ **Note**: The dataset files (`Training.csv` and `Testing.csv`) are **not included in this repository** due to size/licensing restrictions.  

You will need to:
1. Obtain the dataset separately (e.g., the *Kaggle "Disease Prediction from Symptoms"* dataset).  
2. Place them in the project folder with the exact names:
   - `Training.csv`
   - `Testing.csv`

Each dataset contains:
- **Symptom columns** → Binary values (1 = present, 0 = absent)  
- **Target column** → Disease/Prognosis 


## 🚀 Usage
1. Place `Training.csv` and `Testing.csv` in the project directory under the data folder.  
2. Train and run the model with jupyter notebook:
   ```bash
   MLP.ipynb
3. Example conversations
   ```bash
   Describe your symptoms in detail:
   > itching
   It could be Fungal infection or Drug Reaction. Can you share more symptoms?
   > skin rash
   Final prediction: Fungal infection (confidence 0.93)
## 🛠️ Project Structure
- `MLP.py` → Main training + interactive script  
- `Training.csv` → Training dataset  
- `Testing.csv` → Testing dataset  


## 📖 Notes
- Dataset assumes **binary symptom columns** (1 = present, 0 = absent).  
- For **single-symptom inputs**, the model outputs **top-2 possible diseases** and asks for clarification.  
- Dropout and Adam optimizer help reduce **overfitting**.   

---

## 📚 References
- *Neural Networks from Scratch* — S. Rashid (2020)  
- *Adam Optimizer* — Kingma & Ba (2015)  
- *Softmax + Cross Entropy Loss* — Deep Learning literature  
