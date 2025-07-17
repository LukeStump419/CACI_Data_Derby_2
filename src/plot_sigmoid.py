import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend so plots save even in headless mode

print("Starting plot_sigmoid_debug.py")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_basic_sigmoid():
    print("📈 Generating annotated sigmoid curve...")

    z = np.linspace(-10, 10, 1000)
    y = sigmoid(z)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, y, label='sigmoid(z)', linewidth=3, color='darkblue')

    # Add vertical & horizontal centerlines
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)

    # Add shaded background
    ax.fill_between(z, y, 0, where=(z < 0), color='salmon', alpha=0.2, label='Predict: Bad Wine (0)')
    ax.fill_between(z, y, 0, where=(z >= 0), color='lightgreen', alpha=0.2, label='Predict: Good Wine (1)')

    # Annotate key points
    ax.plot([-6], [sigmoid(-6)], 'o', color='maroon')
    ax.text(-6.2, sigmoid(-6)+0.05, 'Very Bad Wine\np ≈ 0.002', fontsize=10)

    ax.plot([0], [0.5], 'o', color='orange')
    ax.text(0.3, 0.52, 'Decision Threshold\np = 0.5', fontsize=10)

    ax.plot([6], [sigmoid(6)], 'o', color='green')
    ax.text(6.1, sigmoid(6)-0.1, 'Very Good Wine\np ≈ 0.998', fontsize=10)

    # Labels & styling
    ax.set_title("How Logistic Regression Predicts Wine Quality", fontsize=16)
    ax.set_xlabel("z = weighted sum of wine features", fontsize=12)
    ax.set_ylabel("Predicted Probability (Good Wine)", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()

    output_path = os.path.join("outputs", "figures", "sigmoid_annotated.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Saved enhanced plot to: {output_path}")
    plt.close()

def main():
    print("Inside main()")
    plot_basic_sigmoid()

if __name__ == "__main__":
    print("Entering __main__ block")
    main()