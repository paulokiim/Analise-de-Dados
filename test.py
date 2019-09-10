#%%
import pandas as pd
import numpy as np

## Aqui acontece a descrição dos dados.
## Printando-os na tela.
df = pd.read_csv("dataset.csv")
#df.drop_duplicates(keep=False, inplace=True)

print (df)
newDf = df.head(10)

groupedByNeighbour = df.groupby(["name", "host_name"])
# groupedByNeighbour["number_of_reviews"].aggregate([np.sum, np.mean, np.std])

#%%
