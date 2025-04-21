# HYDRA-TS: Diverse Representation Learning for Time Series Classification

HYDRA-TS is a feature-based framework for time series classification that systematically integrates multiple mathematical representations. It achieves state-of-the-art accuracy while remaining computationally efficient.

---

## 🚀 Quick Start Guide

### 1. Install Requirements

```bash
pip install numpy pandas scikit-learn aeon tsfresh PyWavelets dtaidistance

2. Load Any UCR Dataset via aeon
HYDRA-TS supports any dataset from the UCR Time Series Archive via the aeon library.

To change the dataset, simply modify the name_dset in the script:
name_dset = "ArrowHead"  # Replace with any UCR dataset name

✅ Example Datasets
"ECG200"
"GunPoint"
"Coffee"
"Plane"
"ItalyPowerDemand"
"Chinatown"
"Beef"

To list all available datasets:
from aeon.datasets import get_dataset_names
print(get_dataset_names("classification"))

3. Run the Demo
python demo_.py
Sample Output:
run:0 acc:0.9302, dur_tr_trans:0.24s, dur_clf_fit:0.12s, dur_te_trans:0.19s

🧠 How It Works
Load a dataset from the UCR archive using aeon.
Extract features using HYDRA-TS from multiple domains (Fourier, Wavelet, Hilbert, etc.).
Train a classifier (e.g., Extremely Randomized Trees).
Evaluate accuracy and compute time.

📁 Project Structure
hydra-ts/
├── demo_.py       # Example: Run on any UCR dataset
├── hydraTS.py     # HYDRA-TS implementation (user-defined)
├── README.md      # This file

⚙️ Customize Feature Extraction
You can control which feature categories are used in hydraTS:
model_feat = hydraTS(
    norms=True,
    stats=True,
    series=True,
    temp=True,
    multiprocessing=True
)

📎 Citation
If you use this code, please cite:
@article{YourHydraTS2024,
  title={HYDRA-TS: Multi-Representation Feature Learning for Efficient Time Series Classification},
  author={Your Name},
  year={2024}
}

📬 Contact
For feedback or questions, open an issue or email celal.alagoz@gmail.com.
