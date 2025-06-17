# ⚡ CFIRE: Cross-Representation Feature Extraction for Time Series Classification

**CFIRE** (Cross-Representation Feature Extraction) is a feature-based time series classification framework that leverages a diverse set of time-domain, frequency-domain, and transformation-based representations. It systematically extracts, combines, and optimizes features across representations to achieve state-of-the-art classification performance.

---

## 🚀 Key Features

- 🔁 Cross-representation feature extraction (Time, Frequency, Derivative, DWT, FFT, Hilbert, etc.)
- 🧠 Support for popular feature sets: [Catch22](https://github.com/chlubba/catch22), [TSFresh](https://github.com/blue-yonder/tsfresh)
- 🔍 Integrated feature redundancy reduction and optimization
- ⚙️ Built-in classifier selection and benchmarking (e.g., ExtraTrees, XGBoost, SVM)
- 🧪 Parallelized feature extraction for scalable performance
- 📊 Comprehensive experimental suite for reproducible research

---

## 📦 Installation

```bash
git clone https://github.com/alagoz/cfire.git
cd cfire
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### 1. Install Requirements

```bash
pip install numpy pandas scikit-learn aeon tsfresh PyWavelets dtaidistance
```

### 2. Load Any UCR Dataset via aeon
HYDRA-TS supports any dataset from the UCR Time Series Archive via the aeon library.

To change the dataset, simply modify the name_dset in the script:
```
name_dset = "ArrowHead"  # Replace with any UCR dataset name
```

✅ Example Datasets  
"ECG200"  
"GunPoint"  
"Coffee"  
"Plane"  
"ItalyPowerDemand"  
"Chinatown"  
"Beef"  

To list all available datasets:
```bash
from aeon.datasets import get_dataset_names
print(get_dataset_names("classification"))
```

### 3. Run the Demo
```
python demo_.py  
```
Sample Output:  
```bash
run:0 acc:0.9535, dur_tr_trans:10.59s, dur_clf_fit:0.31s, dur_te_trans:0.27s, dur_clf_pred:0.02s
run:1 acc:0.9070, dur_tr_trans:0.59s, dur_clf_fit:0.28s, dur_te_trans:0.22s, dur_clf_pred:0.02s
run:2 acc:0.9070, dur_tr_trans:0.52s, dur_clf_fit:0.20s, dur_te_trans:0.27s, dur_clf_pred:0.02s
```

🧠 How It Works  
Load a dataset from the UCR archive using aeon.  
Extract features using CFIRE from multiple domains (Fourier, Wavelet, Hilbert, etc.).  
Train a classifier (e.g., Extremely Randomized Trees).  
Evaluate accuracy and compute time.  

📁 Project Structure  
```bash
cfire/  
├── demo_.py       # Example: Run on any UCR dataset  
├── crossfire.py   # crossfire implementation (user-defined)  
├── README.md      # This file
```

⚙️ Customize Feature Extraction  
You can control which feature categories are used in CFIRE:  
```
model_feat = CFIRE(
    norms=True,
    stats=True,
    series=True,
    temp=True,
    multiprocessing=True
)
```

📬 Contact  
For feedback or questions, open an issue or email celal.alagoz@gmail.com.
