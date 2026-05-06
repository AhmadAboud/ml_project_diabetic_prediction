import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


import pandas as pd
df_5050_01 = pd.read_csv(r"Original\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
alleSpalte = df_5050_01.columns[0:]
print(alleSpalte)


# Definiere die Features für den Plot
x1, y1, z1, color1 = 'BMI', 'Age', 'GenHlth', 'Diabetes_binary'
x2, y2, z2, color2 = 'GenHlth', 'BMI', 'Income', 'Diabetes_binary'

# Diskrete Farben für 0, 1, 2
colors = ['blue',  'red']
cmap_discrete = ListedColormap(colors)

# Grenzen für 0, 1, 2
bounds = [-0.1, 0.1, 1.1]
norm = BoundaryNorm(bounds, cmap_discrete.N)

# Punktgrößen:
# 0 = blau = normal
# 1 = orange = normal
# 2 = rot = doppelt so groß
sizes1 = df_5050_01[color1].map({0: 15, 1: 15, 2: 30})
sizes2 = df_5050_01[color2].map({0: 15, 1: 15, 2: 30})

fig = plt.figure(figsize=(16, 6))

# Plot 1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
sc1 = ax1.scatter(
    df_5050_01[x1],
    df_5050_01[y1],
    df_5050_01[z1],
    c=df_5050_01[color1],
    cmap=cmap_discrete,
    norm=norm,
    s=sizes1,
    alpha=1.0
)
ax1.set_xlabel(x1)
ax1.set_ylabel(y1)
ax1.set_zlabel(z1)
ax1.set_title(f'3D Plot: {x1} vs {y1} vs {z1}')

cbar1 = fig.colorbar(sc1, ax=ax1, ticks=[0, 1, 2])
cbar1.set_label('Diabetes Kategorien')
cbar1.ax.set_yticklabels(['0', '1', '2'])

# Plot 2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
sc2 = ax2.scatter(
    df_5050_01[x2],
    df_5050_01[y2],
    df_5050_01[z2],
    c=df_5050_01[color2],
    cmap=cmap_discrete,
    norm=norm,
    s=sizes2,
    alpha=1.0
)
ax2.set_xlabel(x2)
ax2.set_ylabel(y2)
ax2.set_zlabel(z2)
ax2.set_title(f'3D Plot: {x2} vs {y2} vs {z2}')

cbar2 = fig.colorbar(sc2, ax=ax2, ticks=[0, 1, 2])
cbar2.set_label('Diabetes Kategorien')
cbar2.ax.set_yticklabels(['0', '1', '2'])

plt.tight_layout()
plt.show()