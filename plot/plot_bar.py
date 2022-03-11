import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def to_percent(temp, position):
    return '%1.2f'%(temp)

datasets = ["Actor","Cornell","Texas","Wisconsin"]
x = np.linspace(start=1, stop=4, endpoint=True, num=4, dtype=np.int32)
GraphSAGE = [34.23,75.95,82.43,81.18]
GraphSAGE_JK = [34.28,75.68,83.78,81.96]
NEDA = [38.17,87.78,87.50,89.40]
NEDA_star = [37.80,87.22,88.06,89.80]

fig, ax = plt.subplots(figsize=(15, 6))
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 4

ax.bar(x, GraphSAGE,  width=width, label='GraphSAGE')
ax.bar(x + width, GraphSAGE_JK, width=width, label='GraphSAGE_JK')
ax.bar(x + 2 * width, NEDA, width=width, label='NEDA')
ax.bar(x + 3 * width, NEDA_star, width=width, label='NEDA*')

ax.set_ylabel(r'Test Accuracy(%)', size=22, color='black')
plt.tick_params(labelsize=22)
plt.ylim([30,100])
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(labels=datasets, size=22)
ax.set_yticks(ticks=np.linspace(start=30, stop=100, endpoint=True, num=8, dtype=np.float32))

ax.grid(axis='y', linestyle=':', linewidth=1, color='gray', which='major')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.legend(fontsize = 18)
plt.tight_layout()
plt.savefig(fname="bar.png")
plt.show()

