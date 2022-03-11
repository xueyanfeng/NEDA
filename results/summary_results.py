import numpy as np
import pandas as pd

name = "texas_NEDA_star_{}_{}_{}.csv"

results = []
for s1 in np.linspace(3 ,24 ,8).astype(np.int32):
    for s2 in np.linspace(3 ,24 ,8).astype(np.int32):
        for s0 in np.ceil(np.linspace(max(s1 ,s2) ,2 * max(s1 ,s2) ,5)).astype(np.int32):
            results += pd.read_csv(name.format(s1,s2,s0)).values.tolist()
pd.DataFrame(results,columns = ["s1","s2","s0","accuracy"]).to_csv(name.replace("_{}",""))