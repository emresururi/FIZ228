---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Midterm Exam #2 - Hands On Session Solutions
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

17/05/2024

+++

Classify the 4-dimensional data of 120 points listed in the "FIZ228_20232_MT2_HandsOnExam_data.csv" file via k-means clustering method into 5 classes.

Output should contain:

* The association of each point into their corresponding mean
* Position of the means

Zip your .ipynb file as "FIZ228_20232_MT2_YourName.zip"(*) (don't use Turkish characters as it may result in error!) and upload it to hadi.


(*) this time you don't need to produce an html version (it's a little bit complicated via colab), only the ipynb file is sufficient -- we are actually archiving it because hadi doesn't accept ipynb files.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```{code-cell} ipython3
def kmeans():
    # Initialize the means
    k = 5
    means = np.random.rand(k,4)-0.5
    
    dx_cols = []
    
    while True:
        # Calculate the distances to each mean
        for i in range(1,k+1):
            df["dx"+str(i)] = np.linalg.norm(df.loc[:,["x1","x2","x3","x4"]]-means[i-1,:],axis=1)
            dx_cols.append("dx"+str(i))

        # Find the closest mean
        a = df.loc[:,dx_cols].idxmin(axis=1)

        # Assign the closest mean
        for i in range(1,k+1):
            df.loc[a=="dx"+str(i),"kno"] = i

        # update
        means_new = means.copy()
        for i in range(1,k+1):
            filter_i = df["kno"] == i
            if(not any(filter_i)):
                #print(i,"nope")
                continue
            #print(i,np.sum(filter_i))
            means_new[i-1,:] = df.loc[filter_i,["x1","x2","x3","x4"]].mean(axis=0).to_numpy()

        # Check convergence
        if(np.allclose(means_new,means,atol=1E-6)):
            break
        means = means_new.copy()
        #print("-"*10)
    return means
```

```{code-cell} ipython3
df = pd.read_csv("../data/FIZ228_20232_MT2_HandsOnExam_data.csv",index_col=None)
kmeans()
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3

```
