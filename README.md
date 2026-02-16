# 🧠 Symptom-to-Disease Prediction with Interactive Conversation

A **neural network from scratch** built with **NumPy** to predict diseases from symptoms. The project includes both a standalone training script and an interactive Jupyter notebook, with a chatbot-like interface for symptom-based diagnosis.

## ✨ Features
- **Custom MLP Neural Network** (no deep learning frameworks):
  - Dense layers with He and Xavier initialization
  - ReLU activation functions  
  - Dropout regularization (configurable rate)  
  - Softmax + Cross-Entropy loss  
  - **Adam optimizer** with bias correction and learning rate decay
- **Confidence-based prediction**:
  - Threshold-aware abstention mechanism
  - Selective prediction with coverage metrics
  - Confidence calibration analysis
- **Interactive diagnosis system**:
  - Conversational loop for symptom collection
  - Top-2 disease suggestions when confidence is low
  - Automatic final prediction when confidence threshold is reached
- **Model evaluation**:
  - Stratified k-fold cross-validation (notebook version)
  - Confusion matrix with abstention tracking
  - Calibration curves and confidence histograms
  - Selective accuracy, coverage, and abstain rate metrics

## 📂 Project Structure
```
AI_Disease_Prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── code/
│   └── Main.py                        # Main training + inference script
├── Data/
│   ├── Training.csv                   # Training dataset (not included)
│   └── Testing.csv                    # Testing dataset (not included)
└── notebooks/
    ├── Final_Year_Project.ipynb       # Full pipeline with cross-validation
    └── MLP.ipynb                      # Exploratory notebook
```

## 📥 Dataset
⚠️ **Note**: The dataset files (`Training.csv` and `Testing.csv`) are **not included** due to size/licensing restrictions.

**Required dataset format:**
- Each row represents a patient sample
- **Symptom columns**: Binary values (1 = present, 0 = absent)
- **Target column**: Named `disease` or `prognosis` (disease name/type)

**Obtaining the dataset:**
1. Download from [Kaggle - Disease Prediction from Symptoms](https://www.kaggle.com/datasets)
2. Place files in the `Data/` directory with exact names:
   - `Training.csv`
   - `Testing.csv`

## 🚀 Quick Start

### Option 1: Standalone Training & Inference
```bash
cd code
python Main.py
```
Expected output:
```
Starting training...

Epoch 001 | Train Loss: 2.3451 | Val Acc: 0.6234
Epoch 005 | Train Loss: 1.2345 | Val Acc: 0.8123
...
Optimal abstain threshold: 0.742

Test Results (Selective Prediction)
-----------------------------------
Selective Accuracy : 0.9234
Coverage           : 0.8500
Abstain Rate       : 0.1500
```

### Option 2: Interactive Diagnosis (Jupyter Notebook)
Open `notebooks/Final_Year_Project.ipynb` and run all cells. The notebook includes:
- 5-fold cross-validation on training data
- Optimal threshold tuning
- Interactive diagnosis loop at the end

**Example interaction:**
```
Enter symptoms separated by commas (empty line to exit):
> itching
It could be Fungal infection confidence: 0.65 or Drug Reaction confidence: 0.25. Could you share more symptoms...
> skin rash
It could be Fungal infection confidence: 0.88 or Bacterial infection confidence: 0.10. Could you share more symptoms...
> burning sensation
Final prediction: Fungal infection with confidence: 0.92
```

## 🛠️ Architecture

### MLP Configuration
- **Input Layer**: Variable (depends on symptom count)
- **Hidden Layer 1**: 64 neurons, ReLU activation, 20% Dropout
- **Hidden Layer 2**: 32 neurons, ReLU activation, 20% Dropout
- **Output Layer**: N neurons (softmax), where N = number of diseases

### Training Configuration
- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 64
- **Epochs**: 50-100
- **Validation Split**: 20% of training data
- **Dropout Rate**: 0.2

## 📊 Model Evaluation Metrics
- **Selective Accuracy**: Accuracy on accepted (confident) predictions
- **Coverage**: Ratio of accepted predictions to total predictions
- **Abstain Rate**: Ratio of rejected (low-confidence) predictions
- **Calibration**: Expected vs actual confidence (see calibration curves)

## 📝 Implementation Details

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Neuron` | Main.py | Single neuron with forward/backward pass |
| `Dense` | Main.py | Fully connected layer |
| `ReLU` | Main.py | ReLU activation |
| `Dropout` | Main.py | Regularization during training |
| `SoftmaxCrossEntropy` | Main.py | Combined loss + activation |
| `Adam` | Main.py | Adaptive learning rate optimizer |
| `MLP` | Main.py | Full neural network |
| `LabelEncoder` | Main.py | Encoding/decoding disease labels |

### Training Pipeline (Main.py)
1. **Load & preprocess**: CSV loading, column detection
2. **Encode labels**: Convert disease names to integers
3. **Train/validation split**: 80/20 split with shuffling
4. **Train model**: Batch training with Adam optimizer
5. **Threshold tuning**: Find optimal confidence threshold on validation set
6. **Evaluation**: Test accuracy, confusion matrix, calibration analysis
7. **Visualization**: Confidence histogram and calibration curve plots

## 🔧 Dependencies
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration
Edit parameters in `Main.py`:
```python
# Model architecture
h1 = 64          # Hidden layer 1 size
h2 = 32          # Hidden layer 2 size
dropout = 0.2    # Dropout rate

# Training
epochs = 50      # Number of epochs
batch_size = 64  # Batch size
lr = 0.001       # Learning rate

# Threshold tuning
threshold_range = 0.2 to 0.9  # Confidence thresholds to test
```

## 📈 Output Artifacts
The script generates two visualizations:
- `confidence_histogram.png` - Distribution of model confidence on test set
- `calibration_curve.png` - Expected vs actual accuracy by confidence level
