# Plotting the bar graph with stacked segments and labels
import matplotlib.pyplot as plt
import numpy as np

# Data for counts
categories = ['CE', 'DF', 'GN7 일반', 'GN7 파노라마']
counts = [5562, 5195, 5089, 5393]

# Split ratios
ratios = [0.8, 0.1, 0.1]

# Calculate the data splits
train_counts = [int(count * ratios[0]) for count in counts]
val_counts = [int(count * ratios[1]) for count in counts]
test_counts = [int(count * ratios[2]) for count in counts]

# Bar positions
bar_positions = np.arange(len(categories))

# Plotting
plt.figure(figsize=(10, 6))

bars_train = plt.bar(bar_positions, train_counts, color='#29BDFD', edgecolor='grey', label='Train')
bars_val = plt.bar(bar_positions, val_counts, bottom=train_counts, color='#F46920', edgecolor='grey', label='Val')
bars_test = plt.bar(bar_positions, test_counts, bottom=np.array(train_counts) + np.array(val_counts), color='#F53255', edgecolor='grey', label='Test')

# Adding labels on each segment
for bar in bars_train:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval/2, int(yval), ha='center', va='center', color='white', fontweight='bold')

for bar in bars_val:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, bar.get_y() + yval/2, int(yval), ha='center', va='center', color='white', fontweight='bold')

for bar in bars_test:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, bar.get_y() + yval/2, int(yval), ha='center', va='center', color='white', fontweight='bold')

# Adding labels
plt.xlabel('차종', fontweight='bold')
plt.ylabel('데이터 수', fontweight='bold')
plt.xticks(bar_positions, categories)
plt.title('차종별 데이터')

plt.legend()
plt.show()
