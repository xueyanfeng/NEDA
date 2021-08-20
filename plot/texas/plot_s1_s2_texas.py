from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result = "texas"
ylim = (2,25)
top = 26
color_list = ["blue", "green", "red", "black"]


def get_santter_dots(df,top):
    santter_dots = {}
    df_focus = df.sort_values(by = "accuracy",axis = 0,ascending = False).iloc[:top]
    s0_list = list(set(df_focus["s0"]))
    for i,e in enumerate(s0_list):
        santter_dots[e] = [df_focus[df_focus["s0"] == e]["s1"].tolist(),df_focus[df_focus["s0"] == e]["s2"].tolist(),color_list[i]]
    return dict(sorted(santter_dots.items(), key=lambda item:len(item[1][0]),reverse = True))


fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121)
df_SIR = pd.read_csv("{}_NEDA.csv".format(result))
santter_dots = get_santter_dots(df_SIR,top)
for i,(k,v) in enumerate(santter_dots.items()):
    plt.scatter(v[0],v[1], s=400, label=r"$s_{0}$" + " = {}({})".format(k,len(v[0])), c=color_list[i], marker=str(i+1), alpha=None, edgecolors='white')
plt.title('NEDA',fontsize=22)
plt.xlabel(r'$s_{1}$',fontsize=26)
plt.ylabel(r'$s_{2}$',fontsize=26)
plt.xticks(np.linspace(0,27,10))
plt.yticks(np.linspace(0,27,10))
plt.legend(fontsize = 18,loc = "lower left")
plt.tick_params(labelsize=22)


ax2 = fig.add_subplot(122)
df_SIR_star = pd.read_csv("{}_NEDA_star.csv".format(result))
santter_dots = get_santter_dots(df_SIR_star,top)
for i,(k,v) in enumerate(santter_dots.items()):
    plt.scatter(v[0],v[1], s=400, label=r"$s_{0}$" + " = {}({})".format(k,len(v[0])), c=color_list[i], marker=str(i+1), alpha=None, edgecolors='white')
plt.title('NEDA*',fontsize=22)
plt.xlabel(r'$s_{1}$',fontsize=26)
plt.ylabel(r'$s_{2}$',fontsize=26)
plt.xticks(np.linspace(0,27,10))
plt.yticks(np.linspace(0,27,10))
plt.legend(fontsize = 18,loc = "lower left")
plt.tick_params(labelsize=22)


plt.tight_layout()
plt.savefig(fname="{}_s1_s2.png".format(result))
plt.show()