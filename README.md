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

python
Copy
Edit
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
