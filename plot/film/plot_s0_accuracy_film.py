from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

result = "film"
ylim = (0.355,0.385)
best = 0.8667

def to_percent(temp, position):
    return '%1.1f'%(100*temp)

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121)
df_SIR = pd.read_csv("{}_NEDA.csv".format(result))
ax1.scatter(df_SIR["s0"], df_SIR["accuracy"], marker='o')
x_max = df_SIR.loc[df_SIR["accuracy"] == df_SIR["accuracy"].max()].iloc[0]["s0"].item()
y_max = df_SIR.loc[df_SIR["accuracy"] == df_SIR["accuracy"].max()].iloc[0]["accuracy"].item()
ax1.text(0.5+x_max,y_max-0.0005,"({:.0f}, {:.2f})".format(x_max, 100*y_max),fontsize=18)
x_min = df_SIR.loc[df_SIR["accuracy"] == df_SIR["accuracy"].min()].iloc[0]["s0"].item()
y_min = df_SIR.loc[df_SIR["accuracy"] == df_SIR["accuracy"].min()].iloc[0]["accuracy"].item()
ax1.text(0.5+x_min,y_min-0.0005,"({:.0f}, {:.2f})".format(x_min, 100*y_min),fontsize=18)


plt.title('NEDA',fontsize=22)
plt.ylabel(r'Test Accuracy(%)',fontsize=20)
plt.xticks(np.linspace(0,50,11))
print("(y_min,y_max):({},{})".format(y_min,y_max))
plt.ylim(ylim)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.tick_params(labelsize=18)
top10 = df_SIR.sort_values(by="accuracy",axis=0,ascending=False).loc[9,"accuracy"]
print("top110:{}".format(top10))
plt.axhline(y = top10,
            color="#c72e29",linestyle='--',)
plt.axvline(x = x_max,color = "k",linestyle='--',)
ax1.annotate('top-10', xy=(5, top10), xytext=(7.5, top10 + 0.003),size=18,
            arrowprops=dict(facecolor='g', shrink=0.01))
plt.xlabel(r'$s_{0}$',fontsize=24)

ax2 = fig.add_subplot(122)
df_SIR_star = pd.read_csv("{}_NEDA_star.csv".format(result))
ax2.scatter(df_SIR_star["s0"], df_SIR_star["accuracy"], marker='o')
x_max = df_SIR_star.loc[df_SIR_star["accuracy"] == df_SIR_star["accuracy"].max()].iloc[0]["s0"].item()
y_max = df_SIR_star.loc[df_SIR_star["accuracy"] == df_SIR_star["accuracy"].max()].iloc[0]["accuracy"].item()
ax2.text(0.5+x_max,y_max-0.0005,"({:.0f}, {:.2f})".format(x_max, 100*y_max),fontsize=18)
x_min = df_SIR_star.loc[df_SIR_star["accuracy"] == df_SIR_star["accuracy"].min()].iloc[0]["s0"].item()
y_min = df_SIR_star.loc[df_SIR_star["accuracy"] == df_SIR_star["accuracy"].min()].iloc[0]["accuracy"].item()
ax2.text(0.5+x_min, y_min-0.0005,"({:.0f}, {:.2f})".format(x_min, 100*y_min),fontsize=18)
plt.title('NEDA*',fontsize=22)

plt.ylabel(r'Test Accuracy(%)',fontsize=20)
plt.xticks(np.linspace(0,50,11))
print("(y_min,y_max):({},{})".format(y_min,y_max))
plt.ylim(ylim)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.tick_params(labelsize=18)
top10 = df_SIR_star.sort_values(by="accuracy",axis=0,ascending=False).loc[9,"accuracy"]
print("top110:{}".format(top10))
plt.axhline(y = top10,
            color="#c72e29",linestyle='--',)
plt.axvline(x = x_max,color = "k",linestyle='--',)
ax2.annotate('top-10', xy=(5, top10), xytext=(7.5, top10 + 0.003), size=18,
            arrowprops=dict(facecolor='g', shrink=0.01))
plt.xlabel(r'$s_{0}$',fontsize=24)

plt.tight_layout()
plt.savefig(fname="{}.png".format(result))
plt.show()