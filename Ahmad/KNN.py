import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Damit kann Python höchstens 4 CPUs bzw. Kerne nutzen

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report,
    brier_score_loss, roc_auc_score )
from sklearn.calibration import calibration_curve



# Daten laden
df_5050_01 = pd.read_csv(r"Original\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

alleSpalte = df_5050_01.columns[0:]
# print(alleSpalte)



# Features und Ziel
X = df_5050_01.drop(columns=["Diabetes_binary"])
y = df_5050_01["Diabetes_binary"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# --------------------------------------------------
# 1) Regressor als Vergleich (ohne Skalierung)
# --------------------------------------------------
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

y_pred_cont = knn_reg.predict(X_test)
y_pred_class_reg = (y_pred_cont >= 0.5).astype(int)

print("\n" + "__" * 3, "KNN Regressor ohne Skalierung", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred_class_reg))
print(classification_report(y_test, y_pred_class_reg))
print("MSE:", mean_squared_error(y_test, y_pred_cont))
print("MAE:", mean_absolute_error(y_test, y_pred_cont))
print("\n" + "__" * 50 + "\n")

'''
Bei einem binären Ziel 0/1 heißt MAE ≈ 0.34, dass die Regressionsvorhersage im Mittel etwa 0.34 vom echten Wert entfernt ist;
das ist brauchbar, aber kein besonders niedriger Fehler.
Für MSE und MAE gilt bei scikit-learn grundsätzlich: je kleiner, desto besser, und der beste Wert ist 0.0.
'''


# --------------------------------------------------
# 2) Skalierung für das Hauptmodell
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 3) Hauptmodell: KNeighborsClassifier
# --------------------------------------------------
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)

y_pred = knn_clf.predict(X_test_scaled)
y_proba = knn_clf.predict_proba(X_test_scaled)[:, 1]

print("__" * 3, "KNN Classifier mit Skalierung", "__" * 3)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

brier = brier_score_loss(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print("Brier Score:", brier)
print("ROC-AUC:", roc_auc)

# --------------------------------------------------
# 4) Kalibrierungskurve
# --------------------------------------------------
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

'''
Der Brier Score misst die Qualität probabilistischer Vorhersagen, 
und bei diesem Maß ist 0 ideal; 0.2038 zeigt also eine ordentliche, aber noch verbesserungsfähige Kalibrierung.
Die ROC-AUC misst die Trennschärfe des Modells über alle Schwellenwerte hinweg, und ein Wert von 0.766 bedeutet, 
dass das Modell die beiden Klassen deutlich besser als Zufall trennt, aber noch nicht besonders stark ist.
Ein Brier Score zwischen 0 und 1 wird üblicherweise so gelesen: je näher an 0, desto besser.
Ein ROC-AUC-Wert von 0.766 liegt klar über 0.5, also über dem Zufallsniveau.
'''
