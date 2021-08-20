import matplotlib.pyplot as plt
import pandas as pd

data_SIRSAGE = pd.read_excel("NEDA_enc.xlsx",index_col=0)
data_SIRSAGE_star = pd.read_excel("NEDA_star_enc.xlsx",index_col=0)
fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121)
N, bins, patches = plt.hist(data_SIRSAGE.mean(axis = 1), bins=10, facecolor="blue" ,alpha=0.7)
plt.xlabel(r'$ned$',fontsize=26)
plt.title('NEDA',fontsize=22)
plt.tick_params(labelsize=22)

ax2 = fig.add_subplot(122)
N, bins, patches = plt.hist(data_SIRSAGE_star.mean(axis = 1), bins=10, facecolor="blue" ,alpha=0.7)
plt.xlabel(r'$ned$',fontsize=26)
plt.title('NEDA*',fontsize=22)
plt.tick_params(labelsize=22)

plt.tight_layout()
plt.savefig(fname="hist_film.png")
plt.show()