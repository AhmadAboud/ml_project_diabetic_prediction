import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import tkinter as tk
from tkinter import messagebox
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ---------------------------------------
# Daten laden und Modell trainieren
# ---------------------------------------
df_5050_01 = pd.read_csv(r"Original\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

X = df_5050_01.drop(columns=["Diabetes_binary"])
y = df_5050_01["Diabetes_binary"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)

# Evaluation
y_pred = knn_clf.predict(X_test_scaled)
y_proba = knn_clf.predict_proba(X_test_scaled)[:, 1]

print("__" * 3, "KNN Classifier mit Skalierung", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

brier = brier_score_loss(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print("Brier Score:", brier)
print("ROC-AUC:", roc_auc)

# Optional: Kalibrierungskurve anzeigen
vorhersage_true, vorhersage_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(vorhersage_pred, vorhersage_true, marker="o", label="KNN Classifier")
plt.plot([0, 1], [0, 1], "--", label="Perfekte Kalibrierung")
plt.xlabel("Mittlere vorhergesagte Wahrscheinlichkeit")
plt.ylabel("Anteil der positiven Fälle")
plt.title("Kalibrierungskurve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------
# GUI-Funktion
# ---------------------------------------
def vorhersagen():
    try:
        eingaben = []
        for feature in feature_names:
            wert = float(entries[feature].get())
            eingaben.append(wert)

        neue_daten = pd.DataFrame([eingaben], columns=feature_names)
        neue_daten_scaled = scaler.transform(neue_daten)

        klasse = knn_clf.predict(neue_daten_scaled)[0]
        wahrscheinlichkeit = knn_clf.predict_proba(neue_daten_scaled)[0][1]

        if klasse == 1:
            text = f"Ergebnis: Diabetes\nWahrscheinlichkeit: {wahrscheinlichkeit:.2%}"
        else:
            text = f"Ergebnis: Gesund\nWahrscheinlichkeit für Diabetes: {wahrscheinlichkeit:.2%}"

        result_label.config(text=text, fg="red" if klasse == 1 else "green")

    except ValueError:
        messagebox.showerror("Fehler", "Bitte für alle Felder gültige Zahlen eingeben.")

# ---------------------------------------
# GUI erstellen
# ---------------------------------------
root = tk.Tk()
root.title("Diabetes Vorhersage mit KNN")
root.geometry("700x900")

title_label = tk.Label(root, text="Diabetes-Vorhersage", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

entries = {}

for i, feature in enumerate(feature_names):
    label = tk.Label(frame, text=feature, anchor="w", width=20)
    label.grid(row=i, column=0, padx=5, pady=4, sticky="w")

    entry = tk.Entry(frame, width=20)
    entry.grid(row=i, column=1, padx=5, pady=4)
    entries[feature] = entry

predict_button = tk.Button(root, text="Vorhersage berechnen", command=vorhersagen, bg="lightblue", font=("Arial", 12, "bold"))
predict_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=20)

root.mainloop()