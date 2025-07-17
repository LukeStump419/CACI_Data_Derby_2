import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
import os

print("📈 Generating 3D sigmoid surface...")

# Load training data
df = pd.read_csv("data/train.csv")

# Use two predictors
features = ['alcohol', 'volatile_acidity']
X = df[features].values
y = df['quality'].values

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Build grid for surface
alcohol_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
vol_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
xx, yy = np.meshgrid(alcohol_range, vol_range)
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

# Plot surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, probs, cmap='viridis', alpha=0.8, edgecolor='none')

# Labels
ax.set_title("3D Sigmoid Surface for Wine Prediction")
ax.set_xlabel('Alcohol')
ax.set_ylabel('Volatile Acidity')
ax.set_zlabel('Predicted Probability (Good Wine)')
ax.view_init(elev=30, azim=120)

# Save figure
output_path = "outputs/figures/sigmoid_surface_3d.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()
print(f"✅ Saved 3D plot to: {output_path}")