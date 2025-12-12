import numpy as np
import matplotlib.pyplot as plt
import glob
import os

os.makedirs("../c/code/png_frames", exist_ok=True)

files = sorted(glob.glob("../c/code/frames/y_pred_*.txt"))

# for f in files:
#     data = np.loadtxt(f)
#     y_true = data[:, 0]
#     y_pred = data[:, 1]

#     epoch = int(os.path.basename(f)[8:13])  # extract the 00010 number

#     plt.figure(figsize=(8, 5))
#     plt.plot(y_true, label="Ground Truth", linewidth=2)
#     plt.plot(y_pred, label=f"Prediction (Epoch {epoch})", linewidth=2)
#     plt.legend()
#     plt.title(f"Training Progress — Epoch {epoch}")
#     plt.xlabel("Index")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.tight_layout()
    
#     out_path = f"png_frames/frame_{epoch:05d}.png"
#     plt.savefig(out_path)
#     plt.close()

#     print("Saved:", out_path)


for f in files:
    data = np.loadtxt(f)
    y_true = data[:, 0]
    y_pred = data[:, 1]

    basename = os.path.basename(f)
    name, ext = os.path.splitext(basename)
    epoch_str = name.split('_')[-1]
    epoch = int(epoch_str)

    plt.figure(figsize=(8,5))
    plt.plot(y_true, label="Ground Truth", linewidth=2)
    plt.plot(y_pred, label=f"Prediction (Epoch {epoch})", linewidth=2)
    plt.legend()
    plt.title(f"Training Progress — Epoch {epoch}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../c/code/png_frames/frame_{epoch:05d}.png")
    plt.close()
