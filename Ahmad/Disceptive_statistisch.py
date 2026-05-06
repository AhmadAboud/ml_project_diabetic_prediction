
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from docutils.nodes import title


def process_files_to_excel(files, drop_target=True):
    dfs = {}
    for name, path in files.items():
        df = pd.read_csv(path)

        '''if drop_target:
            if "Diabetes_binary" in df.columns:
                df = df.drop(columns=["Diabetes_binary"])
            elif "Diabetes_012" in df.columns:
                df = df.drop(columns=["Diabetes_012"])
'''
        dfs[name] = df
    return dfs


files = {
"df_full_012": r"archive\diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
"df_full_01": r"archive\diabetes_binary_health_indicators_BRFSS2015.csv",
"df_5050_01": r"archive\diabetes_012_health_indicators_BRFSS2015.csv"
}


dfs = process_files_to_excel(files, drop_target=True)
df_full_012 = dfs["df_full_012"]
df_full_01 = dfs["df_full_01"]
df_5050_01 = dfs["df_5050_01"]

alleSpalte = df_5050_01.columns[0:]
print(alleSpalte)




# 2D-Plot 1: zwei wichtigste Features, drittes als Farbe
x1, y1 = 'BMI', 'Age'
color1 = 'PhysHlth'

# 2D-Plot 2: nächste wichtige Features
x2, y2 = 'Education', 'GenHlth'
color2 = 'Income'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sc1 = ax1.scatter(
    df_5050_01[x1],
    df_5050_01[y1],
    c=df_5050_01[color1],
    cmap='viridis',
    s=20,
    alpha=0.7
)

ax1.set_xlabel(x1)
ax1.set_ylabel(y1)
ax1.set_title(f'2D Plot: {x1} vs {y1}')
fig.colorbar(sc1, ax=ax1, label=color1)

sc2 = ax2.scatter(
    df_5050_01[x2],
    df_5050_01[y2],
    c=df_5050_01[color2],
    cmap='plasma',
    s=20,
    alpha=0.7
)

ax2.set_xlabel(x2)
ax2.set_ylabel(y2)
ax2.set_title(f'2D Plot: {x2} vs {y2}')
fig.colorbar(sc2, ax=ax2, label=color2)

plt.tight_layout()
plt.show()








'''
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(
    Diabetes_012, Income, Education,
    c=Age,
    cmap='viridis',
    s=20
)
ax1.set_xlabel('Diabetes_012')
ax1.set_ylabel('Income')
ax1.set_zlabel('Education')
ax1.set_title('3D Plot')
fig.colorbar(sc1, ax=ax1, label='Age')


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(
    Diabetes_012,HighBP,
    c=Age,
    cmap='viridis',
    s=20
)
ax2.set_xlabel('Diabetes_012')
ax2.set_ylabel('HighBP')
ax2.set_zlabel('PhysActivity')
ax2.set_title('3D Plot ')
fig.colorbar(sc2, ax=ax2, label='Age')

plt.tight_layout()
plt.show()'''


'''
df_full_012.hist(bins=15, figsize=(14, 9), edgecolor="red")
df_full_01.hist(bins=15, figsize=(14, 9), edgecolor="green")
df_5050_01.hist(bins=15, figsize=(14, 9), edgecolor="blue")
plt.tight_layout()
plt.show()

'''

#https://github.com/NilsBauch/ml_project_diabetic_prediction

# Toreten Diagramm
'''
last5 = df_5050_01.columns[-5:]

fig, axes = plt.subplots(3, 3, figsize=(20, 4))
axes = axes.flatten()

for i, col in enumerate(last5):
    counts = pd.cut(df_5050_01[col], bins=5).value_counts().sort_index()
    axes[i].pie(counts, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    axes[i].set_title(f"Feature {col}")

plt.tight_layout()
plt.show()
'''


'''
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(GenHlth, MentHlth, Education, cmap='viridis', s=20)
ax1.set_xlabel('GenHlth')
ax1.set_ylabel('MentHlth')
ax1.set_zlabel('Education')
ax1.set_title('3D Plot GenHlth MentHlth Education')


ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(HighBP, BMI, Age, cmap='viridis', s=20)
ax2.set_xlabel('HighBP')
ax2.set_ylabel('BMI')
ax2.set_zlabel('Age')
ax2.set_title('3D Plot HighBP BMI Age')

plt.tight_layout()
plt.show()
'''


# from sklearn.model_selection import train_test_split
'''
print(df_full_012.info())
for i in range(len(df_full_012)):
    for j in range(len(df_full_012)):
        df_full_012.plot.scatter(x=df_full_012.columns[i], y=df_full_012.columns[j], alpha=0.5)
        plt.show()
'''



'''

Diabetes_012 = df_5050_01['Diabetes_012']
HighBP = df_5050_01['HighBP']
HighChol = df_5050_01['HighChol']
CholCheck = df_5050_01['CholCheck']
BMI = df_5050_01['BMI']
Smoker = df_5050_01['Smoker']
Stroke = df_5050_01['Stroke']
HeartDiseaseorAttack = df_5050_01['HeartDiseaseorAttack']
PhysActivity = df_5050_01['PhysActivity']
Fruits = df_5050_01['Fruits']
Veggies = df_5050_01['Veggies']
HvyAlcoholConsump = df_5050_01['HvyAlcoholConsump']
AnyHealthcare = df_5050_01['AnyHealthcare']
NoDocbcCost = df_5050_01['NoDocbcCost']
GenHlth = df_5050_01['GenHlth']
MentHlth = df_5050_01['MentHlth']
PhysHlth = df_5050_01['PhysHlth']
DiffWalk = df_5050_01['DiffWalk']
Sex = df_5050_01['Sex']
Age = df_5050_01['Age']
Education = df_5050_01['Education']
Income = df_5050_01['Income']

'''