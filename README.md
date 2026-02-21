# 🧠 Symptom-to-Disease Prediction with Interactive Conversation

A **neural network from scratch** built with **NumPy** to predict diseases from symptoms. The project features a custom MLP implementation with an interactive conversational diagnosis system, no deep learning frameworks required.

## ✨ Features
- **Custom MLP Neural Network** :
  - Fully connected Dense layers with He and Xavier weight initialization
  - ReLU activation with gradient backpropagation
  - Dropout regularization (20% rate) for training stability
  - Softmax + Cross-Entropy loss function
  - Adam optimizer with adaptive learning rates, momentum, and RMSprop
- **Dual-threshold Prediction Strategy**:
  - **Entropy threshold**: Requests additional symptoms when uncertainty is high
  - **Confidence threshold**: Abstains when max probability is below threshold
  - **Calibrated decision-making**: Combines both thresholds for robust predictions
- **Interactive Diagnosis System**:
  - Natural language symptom parsing with comma/period/semicolon delimiters
  - Real-time entropy and confidence tracking during conversation
- **Comprehensive Model Evaluation**:
  - Train/validation split (80/20) with random shuffling
  - Threshold tuning on validation set to maximize accuracy × coverage
  - Test set evaluation with accuracy metrics

## 📂 Project Structure
```
AI_Disease_Prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry point
├── core/
│   └── core_pipeline.py               # Core pipeline utilities
├── Data/
│   ├── Training.csv                   # Training dataset (not included)
│   └── Testing.csv                # Exploratory notebook
├── model/
│   ├── layers.py                      # Layer implementations
│   ├── loss.py                        # Loss functions
│   ├── mlp.py                         # MLP model
│   ├── neuron.py                      # Neuron implementation
│   └── optimization.py                # Optimizers
├── utils/
│   ├── data_utils.py                  # Data loading utilities
│   ├── encoding.py                    # Label encoding
│   └── metrics.py                     # Evaluation metrics
├── interaction/
│   └── conversation.py                # Conversational interface
├── thresholds/
│   └── threshold.py                   # Threshold tuning utilities
└── visualization/
    └── plots.py                       # Visualization functions
```

## 📥 Dataset
⚠️ **Note**: The dataset files (`Training.csv` and `Testing.csv`) are **not included** due to size/licensing restrictions.

**Required dataset format:**
- Each row represents a patient sample
- **Symptom columns**: Binary values (1 = present, 0 = absent)

**Obtaining the dataset:**
1. Download from Kaggle
2. Place files in the `Data/` directory with exact names:
   - `Training.csv`
   - `Testing.csv`

## 🚀 Quick Start

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 1: From Root Directory
```bash
python main.py
```

This runs the core pipeline which loads data, trains the model, evaluates on test set, and starts the interactive diagnosis mode.

**Expected output:**
```
Starting training...

Epoch 001 | Train Acc: 0.6234 | Val Acc: 0.5123 | Loss: 2.3451
Epoch 005 | Train Acc: 0.8234 | Val Acc: 0.8123 | Loss: 1.2345
Epoch 010 | Train Acc: 0.9012 | Val Acc: 0.8756 | Loss: 0.5234
...

Confidence threshold: 0.742
Entropy threshold: 0.568

Test accuracy: 0.8456

Enter symptoms (reset / empty to exit)

Symptoms: itching, skin_rash
Uncertain (entropy=0.95). Possible: Fungal infection (0.65)

Symptoms: nodal_skin_eruptions
Final Diagnosis: Fungal infection (confidence=0.92, entropy=0.42)
```