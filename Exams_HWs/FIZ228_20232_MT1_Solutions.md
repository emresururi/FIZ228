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

# Midterm Exam #1 Solutions
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

19/04/2024

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
```

# 1 
Import the given `FIZ228_20232_MT1_data.csv` file as a pandas dataframe.

This data is known to contain signals taken from two joint sinusoidal signal generators (where superposition applies, i.e., $y = y_1 + y_2$). 

Rename the first column as "t" and the second as "y"

```{code-cell} ipython3
data = pd.read_csv("../data/FIZ228_20232_MT1_data.csv",header=None)
data.columns = ["t","y"]
data
```

# 2

+++

Plot its graph as using red dots (no lines).

```{code-cell} ipython3
plt.plot(data["t"],data["y"],"r.")
plt.show()
```

# 3

As mentioned above, this data consists of the superposition of two sinusoidal signals, such that:

$$y = A_1\sin(\omega_1 t + \varphi_1)+A_2\sin(\omega_2 t + \varphi_2)$$

Determine the parameters (amplitude, angular frequency and phase) of these two sinusoidal signals that best match to the data given.

(Hint: Depending on your method, you can relax the default tolerance, in other words, don't get upset if a very high precision match can't be obtained)

```{code-cell} ipython3
def f(t,params):
    (A1,A2,w1,w2,ph1,ph2) = params
    return A1*np.sin(w1*t+np.deg2rad(ph1)) + \
           A2*np.sin(w2*t+np.deg2rad(ph2))
```

```{code-cell} ipython3
def err_f(params):
    return ((data["y"] - f(data["t"],params))**2).sum()
```

```{code-cell} ipython3
res = opt.minimize(err_f,[3,2,2,1,0,-1])
res
```

# 4

Using these parameters, construct your estimation function and introduce a third column labeled "e" on your dataframe that holds the function values with respect to your t values.

```{code-cell} ipython3
data["e"] = f(data["t"],res.x)
```

```{code-cell} ipython3
data
```

# 5

Plot the given data against the model estimation (data points: red dots as above; model estimation: dashed line)

```{code-cell} ipython3
plt.plot(data["t"],data["e"],"b--")
plt.plot(data["t"],data["y"],"r.")
plt.show()
```

# 6

Calculate the following error estimations between the given data and your model:$\DeclareMathOperator\erf{erf}$

 * $\erf_1 = \sqrt{\sum_{i}{(y_i - e_i)^2}}$
 * $\erf_2 = \sum_{i}{|y_i - e_i|}$
 
 where $y$ indicates the given data and $e$ the model estimation.
 
 Also calculate the coefficient of determination ($r^2$).

```{code-cell} ipython3
erf1 = np.sqrt(((data["y"] - data["e"])**2).sum())
erf2 = np.abs((data["y"] - data["e"]).sum())
erf1,erf2
```

```{code-cell} ipython3
S_r = ((data["y"] - data["e"])**2).sum()
S_t = ((data["y"] - data["y"].mean())**2).sum()
r2 = (S_t - S_r) / S_t

r2
```

# 7

The data presented was obtained by adding noise to the data calculated with the following parameters:

$$\begin{align*}A_1 &= 2.0&
\omega_1 &= 1.0&
\varphi_1 &=0.0\\
A_2 &= 1.5&
\omega_2 &= 2.3&
\varphi_2 &=-35.0^o
\end{align*}$$

_(You can't use these values to spot a good starting position for the above operations!)_


Plot the data along with your fit and the model constructed with these parameters like you did in part 5.

Calculate the error estimations between the data and a model constructed with these parameters like you did in part 6.

```{code-cell} ipython3
params_org = [2.0,1.5,1.0,2.3,0.0,-35.0]
y_org = f(data["t"],params_org)
```

```{code-cell} ipython3
plt.plot(data["t"],y_org,"k-")
plt.plot(data["t"],data["e"],"b--")
plt.plot(data["t"],data["y"],"r.")
plt.legend(["original","fit","data"])
plt.show()
```

```{code-cell} ipython3
erf1o = np.sqrt(((data["y"] - y_org)**2).sum())
erf2o = np.abs((data["y"] - y_org).sum())
erf1o,erf2o
```

```{code-cell} ipython3

```
