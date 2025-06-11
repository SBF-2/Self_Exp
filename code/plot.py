import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/feisong/Desktop/modeltrain/modeltraining/code/log/train_loss.csv')

# Create the plot
plt.figure(figsize=(10, 6))
if df.shape[1] >= 2:  # Ensure there are at least two columns
	plt.plot(df.iloc[:, 0], df.iloc[:, 1], 'b-', linewidth=1)
else:
	raise ValueError("The CSV file does not have enough columns for plotting.")

# Customize the plot
plt.title('Training Loss Over Steps')
plt.xlabel('Steps')
plt.ylabel('Average Loss')
plt.grid(True)

# Save the plot
plt.savefig('training_loss_plot.png')
plt.close()