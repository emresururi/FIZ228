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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

```{code-cell} ipython3
def Gauss2D(mus,sigmas,N):
    data = np.zeros((N,2))
    data[:,0] = np.random.normal(mus[0],sigmas[0],(N))
    data[:,1] = np.random.normal(mus[1],sigmas[1],(N))
    return data
```

```{code-cell} ipython3
def GaussnD(mus,sigmas,N):
    n = len(mus)
    data = np.zeros((N,n))
    for i in range(n):
        data[:,i] = np.random.normal(mus[i],sigmas[i],(N))
    return data
```

```{code-cell} ipython3
mus = [2,3]
sigmas = [0.2,0.2]
N = 30
data2 = GaussnD(mus,sigmas,N)
```

```{code-cell} ipython3
plt.plot(data2[:,0],data2[:,1],"o")
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()
```

```{code-cell} ipython3
data4_1 = GaussnD([1,1,2,3],[1.1,.2,1.2,.4],30)
data4_2 = GaussnD([4,2,0,2],[.6,.4,.3,1.1],20)
data4_3 = GaussnD([0,5,1,3],[.6,1.2,.4,.2],30)
data4_4 = GaussnD([2,5,3,1],[.3,1.1,.6,2.1],25)
data4_5 = GaussnD([3,2,4,5],[.8,.5,.3,.3],15)
data_all = np.concatenate((data4_1,data4_2,data4_3,data4_4,data4_5),axis = 0)
```

```{code-cell} ipython3
data_all.shape
```

```{code-cell} ipython3
df = pd.DataFrame(data_all,columns=["x1","x2","x3","x4"])
df
```

```{code-cell} ipython3
#df.to_csv("../data/FIZ228_20232_MT2_HandsOnExam_data_ordered.csv",index=False)
```

```{code-cell} ipython3
# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("../data/FIZ228_20232_MT2_HandsOnExam_data.csv",index=False)
```

```{code-cell} ipython3
N=4
a = np.zeros((N,2))
a[0,:]
```

```{code-cell} ipython3
d=Gauss2D([2,4],[1,.2],100)
#d
```

```{code-cell} ipython3
plt.plot(d[:,0],d[:,1],"o")
plt.xlim(-5,5)
plt.ylim(-5,10)
plt.show()
```

```{code-cell} ipython3
data1 = Gauss2D([mu1x,mu1y],[sigma1x,sigma1y],N1)
data2 = Gauss2D([mu2x,mu2y],[sigma2x,sigma2y],N2)
data3 = Gauss2D([mu3x,mu3y],[sigma3x,sigma3y],N3)
```

```{code-cell} ipython3
data1.shape
```

```{code-cell} ipython3
plt.plot(data1[:,0],data1[:,1],"o")
plt.plot(data2[:,0],data2[:,1],"x")
plt.plot(data3[:,0],data3[:,1],"^")
plt.xlim(-10,15)
plt.ylim(-10,10)
plt.show()
```

```{code-cell} ipython3
plt.plot(data1[:,0],data1[:,1],"ko",markerfacecolor="None")
plt.plot(data2[:,0],data2[:,1],"ko",markerfacecolor="None")
plt.plot(data3[:,0],data3[:,1],"ko",markerfacecolor="None")
plt.xlim(-10,12)
plt.ylim(-10,8)
plt.show()
```

```{code-cell} ipython3
data_all = np.concatenate((data1,data2,data3),axis=0)
data_all
```

```{code-cell} ipython3
plt.plot(data_all[:,0],data_all[:,1],"ko",markerfacecolor="None")
plt.xlim(-10,12)
plt.ylim(-10,8)
plt.show()
```

```{code-cell} ipython3
DF = pd.DataFrame(data_all)
DF
```

```{code-cell} ipython3
DF.to_csv("/tmp/FIZ228_20232_MT2_WrittenExam_data.csv")
```

```{code-cell} ipython3

```
