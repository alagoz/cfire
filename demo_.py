from sklearn import metrics
from sklearn.model_selection import train_test_split
from crossfire import CFIRE 
from sklearn.ensemble import ExtraTreesClassifier
from time import time
from aeon.datasets import load_classification  # Load datasets from aeon

# ====== Configuration ======
n_resample = 3                      # Number of Monte Carlo resamples
eval_metric = 'acc'                # Evaluation metric
rseed = None                       # Random seed (set to int for reproducibility)

# ====== Load Dataset from UCR Archive via aeon ======
# Replace 'ArrowHead' with any dataset name available in the UCR archive
# Example names: "ECG200", "GunPoint", "Plane", "Coffee", etc.
name_dset = "ArrowHead"
X, y = load_classification(name=name_dset)
X = X.squeeze()

# ====== Define Classifier ======
# Any classifier can be used. Here, Extremely Randomized Trees (ET) are selected
clf = ExtraTreesClassifier(n_estimators=200, random_state=rseed)

# ====== Initialize HYDRA-TS Model ======
# Enable/disable feature categories (norms, stats, series, temporal)
model_feat = CFIRE(
    norms=True,
    stats=True,
    series=True,
    temp=True,
    multiprocessing=True  # Enable multiprocessing for faster transformations
)

# ====== Monte Carlo Cross-Validation ======
for r_ in range(n_resample):
    # Split train/test with stratified sampling
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=r_, stratify=y
    )

    # === Train: Transform ===
    t0 = time()
    x_tr_trans = model_feat.fit_transform(x_tr)
    dur_tr_trans = time() - t0

    # === Train: Fit Classifier ===
    t0 = time()
    clf.fit(x_tr_trans, y_tr)
    dur_fit = time() - t0

    # === Test: Transform ===
    t0 = time()
    x_te_trans = model_feat.fit_transform(x_te)
    dur_te_trans = time() - t0

    # === Predict and Evaluate ===
    t0 = time()
    y_pred = clf.predict(x_te_trans)
    y_pred_proba = clf.predict_proba(x_te_trans)
    dur_pred = time() - t0
    acc_ = metrics.accuracy_score(y_te, y_pred)

    print(
        f'run:{r_} acc:{acc_:.4f}, '
        f'dur_tr_trans:{dur_tr_trans:.2f}s, '
        f'dur_clf_fit:{dur_fit:.2f}s, '
        f'dur_te_trans:{dur_te_trans:.2f}s, '
        f'dur_clf_pred:{dur_pred:.2f}s'
    )
