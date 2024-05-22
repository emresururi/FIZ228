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
mu1x = 2
mu1y = 3
sigma1x = 2
sigma1y = 1
N1 = 20
```

```{code-cell} ipython3
mu2x = -5
mu2y = -2.5
sigma2x = 1.2
sigma2y = 1.2
N2 = 10
```

```{code-cell} ipython3
mu3x = 7.5
mu3y = -5
sigma3x = 1.6
sigma3y = 0.8
N3 = 30
```

```{code-cell} ipython3
def Gauss2D(mus,sigmas,N):
    data = np.zeros((N,2))
    data[:,0] = np.random.normal(mus[0],sigmas[0],(N))
    data[:,1] = np.random.normal(mus[1],sigmas[1],(N))
    return data
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
