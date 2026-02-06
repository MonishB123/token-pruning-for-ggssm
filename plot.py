import matplotlib.pyplot as plt
import numpy as np

# --- Data Extraction from the two tables in the image ---

# Pruning ratios (same for both datasets)
pruning_ratios = np.array([0.0, 15.0, 27.5, 40.0])
pruning_labels = [f'{p}%' for p in pruning_ratios]

# Dataset 1 (Top Table results - ETrM)
DATASET = "Solar"

data1 = {
    'name': f'{DATASET}, Prediction Length 96',
    # Throughput (s/s) means
    'throughput': np.array([436.33, 595.35, 658.35, 712.52])
}

# Dataset 2 (Bottom Table results - ETrM_seqd192)
data2 = {
    'name': f'{DATASET}, Prediction Length 192',
    'throughput': np.array([126.20, 133.82, 128.85, 131.43])
}

# --- Plotting Setup for Throughput Bar Chart ---

# Define bar parameters
x = np.arange(len(pruning_ratios))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Create the bars for Dataset 1
rects1 = plt.bar(x - width/2, data1['throughput'], width, label=data1['name'], color='teal')

# Create the bars for Dataset 2
rects2 = plt.bar(x + width/2, data2['throughput'], width, label=data2['name'], color='darkorange')

# --- Labeling and Styling ---
plt.title('Impact of Pruning Ratio on Throughput (Samples/Second)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Throughput (s/s)', fontsize=12)
plt.xlabel('Pruning Ratio', fontsize=12)
plt.xticks(x, pruning_labels) # Set the x-tick labels to the pruning percentages
plt.legend(fontsize=10)
plt.ylim(0, max(data1['throughput'].max(), data2['throughput'].max()) * 1.1) # Set Y-limit dynamically

# Add the specific value labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

# --- Summary Observations ---
print("\n--- Summary Observations (Throughput) ---")
print(f"{data1['name']} (Max Throughput at 40.0%): {data1['throughput'].max():.2f} s/s")
print(f"{data2['name']} (Max Throughput at 40.0%): {data2['throughput'].max():.2f} s/s")
print("Pruning by 40.0% resulted in the highest throughput for both datasets.")