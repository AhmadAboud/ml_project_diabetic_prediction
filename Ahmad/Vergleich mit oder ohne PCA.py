import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report,
    brier_score_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve

# Daten laden
df_5050_01 = pd.read_csv(r"Original\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Features und Ziel
X = df_5050_01.drop(columns=["Diabetes_binary"])
y = df_5050_01["Diabetes_binary"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================================================
# 1) OHNE PCA
# ==================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regressor ohne PCA
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
y_pred_cont_reg = knn_reg.predict(X_test_scaled)
y_pred_class_reg = (y_pred_cont_reg >= 0.5).astype(int)

print("\n" + "__" * 3, "KNN Regressor ohne PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_class_reg))
print(classification_report(y_test, y_pred_class_reg))
print("MSE:", mean_squared_error(y_test, y_pred_cont_reg))
print("MAE:", mean_absolute_error(y_test, y_pred_cont_reg))

# Classifier ohne PCA
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_clf = knn_clf.predict(X_test_scaled)
y_proba_clf = knn_clf.predict_proba(X_test_scaled)[:, 1]

brier_no_pca = brier_score_loss(y_test, y_proba_clf)
roc_auc_no_pca = roc_auc_score(y_test, y_proba_clf)

print("\n" + "__" * 3, "KNN Classifier ohne PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_clf))
print(classification_report(y_test, y_pred_clf))
print("Brier Score:", brier_no_pca)
print("ROC-AUC:", roc_auc_no_pca)

# ==================================================
# 2) MIT PCA
# ==================================================
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nAnzahl PCA-Komponenten:", pca.n_components_)
print("Erklärte Varianz gesamt:", pca.explained_variance_ratio_.sum())

# Regressor mit PCA
knn_reg_pca = KNeighborsRegressor(n_neighbors=5)
knn_reg_pca.fit(X_train_pca, y_train)
y_pred_cont_reg_pca = knn_reg_pca.predict(X_test_pca)
y_pred_class_reg_pca = (y_pred_cont_reg_pca >= 0.5).astype(int)

print("\n" + "__" * 3, "KNN Regressor mit PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_class_reg_pca))
print(classification_report(y_test, y_pred_class_reg_pca))
print("MSE:", mean_squared_error(y_test, y_pred_cont_reg_pca))
print("MAE:", mean_absolute_error(y_test, y_pred_cont_reg_pca))

# Classifier mit PCA
knn_clf_pca = KNeighborsClassifier(n_neighbors=5)
knn_clf_pca.fit(X_train_pca, y_train)
y_pred_clf_pca = knn_clf_pca.predict(X_test_pca)
y_proba_clf_pca = knn_clf_pca.predict_proba(X_test_pca)[:, 1]

brier_pca = brier_score_loss(y_test, y_proba_clf_pca)
roc_auc_pca = roc_auc_score(y_test, y_proba_clf_pca)

print("\n" + "__" * 3, "KNN Classifier mit PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_clf_pca))
print(classification_report(y_test, y_pred_clf_pca))
print("Brier Score:", brier_pca)
print("ROC-AUC:", roc_auc_pca)

# ==================================================
# 3) Kalibrierungskurven vergleichen
# ==================================================
true_no_pca, pred_no_pca = calibration_curve(y_test, y_proba_clf, n_bins=10)
true_pca, pred_pca = calibration_curve(y_test, y_proba_clf_pca, n_bins=10)

plt.figure(figsize=(7, 7))
plt.plot(pred_no_pca, true_no_pca, marker="o", label="Classifier ohne PCA")
plt.plot(pred_pca, true_pca, marker="o", label="Classifier mit PCA")
plt.plot([0, 1], [0, 1], "--", label="Perfekte Kalibrierung")
plt.xlabel("Mittlere vorhergesagte Wahrscheinlichkeit")
plt.ylabel("Anteil der positiven Fälle")
plt.title("Kalibrierungskurve: ohne PCA vs. mit PCA")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()