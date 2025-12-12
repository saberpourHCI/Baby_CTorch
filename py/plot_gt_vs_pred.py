import matplotlib.pyplot as plt
import numpy as np

# Load the data from C
data = np.loadtxt("../c/plot_data.txt")

y_true = data[:, 0]
y_pred = data[:, 1]

# Plot both lines
plt.figure(figsize=(8, 5))
plt.plot(y_true, label="Ground Truth", linewidth=2)
plt.plot(y_pred, label="Prediction", linewidth=2)

plt.legend()
plt.title("Model Output vs Ground Truth")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()
