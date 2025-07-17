from src.data_loader import load_train_data
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("🔍 Running quick EDA on data/train.csv...\n")

# Load and clean training data
df = load_train_data(return_X_y=False)

# Add label column for visualization
df["quality_label"] = df["quality"].map({0: "Bad", 1: "Good"})
palette = {"Good": "green", "Bad": "red"}

# Print high-level EDA summary
print(f"📏 Data shape: {df.shape}\n")
print("🧠 Columns:")
print(df.columns.tolist(), "\n")

print("📊 Data types:")
print(df.dtypes, "\n")

print("🔍 Missing values:")
print(df.isnull().sum(), "\n")

print("⚖️ Class distribution (quality):")
print(df["quality"].value_counts(), "\n")

print("📈 Descriptive statistics:")
print(df.describe(), "\n")

# Make sure output directory exists
os.makedirs("outputs/figures", exist_ok=True)

# Create the pairplot with transparency
print("📸 Creating histogram matrix (pairplot)...")
sns_plot = sns.pairplot(
    df.drop(columns=["id"]),
    hue="quality_label",
    palette=palette,
    plot_kws={'alpha': 0.5, 's': 20},
    diag_kws={'alpha': 0.8, 'bins': 20},
    diag_kind="hist",
    corner=True
)

output_path = "outputs/figures/eda_pairplot.png"
sns_plot.savefig(output_path, dpi=300)
print(f"✅ Updated pairplot saved to: {output_path}")
