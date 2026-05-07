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

# Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA: z.B. 95% Varianz behalten
pca = PCA(n_components=0.50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Anzahl PCA-Komponenten:", pca.n_components_)
print("Erklärte Varianz gesamt:", pca.explained_variance_ratio_.sum())

# --------------------------------------------------
# 1) Regressor als Vergleich auf PCA-Daten
# --------------------------------------------------
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_pca, y_train)

y_pred_cont = knn_reg.predict(X_test_pca)
y_pred_class_reg = (y_pred_cont >= 0.5).astype(int)

print("\n" + "__" * 3, "KNN Regressor mit PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_class_reg))
print(classification_report(y_test, y_pred_class_reg))
print("MSE:", mean_squared_error(y_test, y_pred_cont))
print("MAE:", mean_absolute_error(y_test, y_pred_cont))

print("\n" + "__" * 50 + "\n")

# --------------------------------------------------
# 2) Hauptmodell: KNeighborsClassifier auf PCA-Daten
# --------------------------------------------------
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_pca, y_train)

y_pred = knn_clf.predict(X_test_pca)
y_proba = knn_clf.predict_proba(X_test_pca)[:, 1]

print("__" * 3, "KNN Classifier mit PCA", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

brier = brier_score_loss(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print("Brier Score:", brier)
print("ROC-AUC:", roc_auc)

# Kalibrierungskurve
vorhersage_true, vorhersage_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(vorhersage_pred, vorhersage_true, marker="o", label="KNN Classifier mit PCA")
plt.plot([0, 1], [0, 1], "--", label="Perfekte Kalibrierung")
plt.xlabel("Mittlere vorhergesagte Wahrscheinlichkeit")
plt.ylabel("Anteil der positiven Fälle")
plt.title("Kalibrierungskurve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()