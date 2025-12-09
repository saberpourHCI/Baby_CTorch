import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../c/loss_data.csv")
plt.plot(data['epoch'], data['loss'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid(True)
plt.show()
