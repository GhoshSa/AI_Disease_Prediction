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
- **Case-Based Verification System**:
  - Secondary validation layer using historical case similarity (K-NN)
  - Logic to "Accept", "Reject", or mark predictions as "Uncertain"
  - Support for "Force" mode to bypass standard entropy checks
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
│   └── Testing.csv                    # Evaluation dataset (not included)
├── model/
│   ├── layers.py                      # Layer implementations
│   ├── loss.py                        # Loss functions
│   ├── mlp.py                         # MLP model
│   ├── neuron.py                      # Neuron implementation
│   └── optimization.py                # Optimizer
├── utils/
│   ├── data_utils.py                  # Data loading utilities
│   ├── encoding.py                    # Label encoding
│   ├── metrics.py                     # Evaluation metrics (Entropy, Accuracy)
│   └── persistence.py                 # Model saving/loading logic
├── interaction/
│   └── conversation.py                # Conversational interface
├── thresholds/
│   ├── threshold.py                   # Standard threshold tuning
│   └── verifier_thresholds.py         # Verifier-specific tuning
├── verification/
│   └── verifier.py                    # Case-based verification logic
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

## 📜 License

This project is for **academic purposes only**.  
Not intended for real-world medical use.

---

## ⭐ Acknowledgements

- Open medical datasets  
- Academic references on ML in healthcare  
- Inspiration from hybrid AI systems  