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

# Final Exam (20232)
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

05/06/2024

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
```

# 1 

Consider the data given in the {download}`FIZ228_20232_Final_data.csv<../data/FIZ228_20232_Final_data.csv>` file. It contains 100 measurements taken for various $x$ values.

Two models have been suggested for the mechanism behind the event: Gaussian and Lorentzian, defined respectively as:

$$\mathcal{G}(x;\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]$$

$$\mathcal{L}(x;x_0,\gamma)=\frac{1}{\pi}\left[\frac{\gamma}{(x-x_0)^2+\gamma^2}\right]$$

Fit the data into these two models, calculate the errors and decide the best model for the data by stating your justification.

**Bonus:** Can you come up with a better model? _(Not talking about a polynomial fit of 100. order!)_ If you can, propose your model, fit it and calculate the error. (Hint, plotting the optimized Gaussian, Lorentzian and the data might give you an idea ;)

+++

## Solution

```{code-cell} ipython3
def G(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(x-mu)**2/sigma**2)
def L(x,x0,A):
    return 1/(np.pi)*A/((x-x0)**2+A**2)
```

```{code-cell} ipython3
data = pd.read_csv("../data/FIZ228_20232_Final_data.csv",header=None)
data.columns = ["x","y"]
data
```

```{code-cell} ipython3
opt_G = opt.curve_fit(G,data.x,data.y,[5,2])
opt_G[0]
```

```{code-cell} ipython3
opt_L = opt.curve_fit(L,data.x,data.y,[5,2])
opt_L[0]
```

```{code-cell} ipython3
def f_errG(ms):
    return np.linalg.norm(G(data.x,ms[0],ms[1]) - data.y)
def f_errL(x0A):
    return np.linalg.norm(L(data.x,x0A[0],x0A[1]) - data.y)
```

```{code-cell} ipython3
f_errG(opt_G[0])
```

```{code-cell} ipython3
f_errL(opt_L[0])
```

```{code-cell} ipython3
opt.minimize(f_errG,[5,2])
```

```{code-cell} ipython3
opt.minimize(f_errL,[5,2])
```

```{code-cell} ipython3
plt.plot(data.x,data.y,"ro")
plt.plot(data.x,G(data.x,opt_G[0][0],opt_G[0][1]),"g-")
plt.plot(data.x,L(data.x,opt_L[0][0],opt_L[0][1]),"b-")
plt.show()
```

# 2

+++

Solve the following ODE for the given conditions:

$$y''-y'-2y=e^{3x}\\
y(0) = 1.8500,\quad y(1) = 25.9714,\quad x\in[0,1];\quad h\le0.01$$

**Bonus:**

Its analytical solution is: $y(x) = -1.3e^{-x}+2.9e^{2x}+\frac{1}{4}e^{3x}$


Calculate the total RMS error defined as:

$$\sqrt{\frac{1}{N}\sum_i{(y_i-t_i)^2}}$$

where $y$ is the model estimate and $t$ is the analytical solution's value (_i.e._, true value).

+++

## Solution

```{code-cell} ipython3
y_0   = 1.85000000
y_Nm1 = 25.9714

N = 105

x = np.linspace(0,1,N)
h = x[1] - x[0]

print(h)
```

```{code-cell} ipython3
A = np.zeros((N-2,N-2))

for i in range(1,N-1):
    A[i-1,i-1] = -4*(1+h**2)
for i in range(1,N-2):
    A[i-1,i] = 2-h
for i in range(2,N-1):
    A[i-1,i-2] = 2+h
    
b = 2*h**2*np.exp(3*x[1:-1]).reshape(N-2,1)
b[0,0] += -(2+h)*y_0
b[-1,0] += -(2-h)*y_Nm1
```

```{code-cell} ipython3
f1_n = np.linalg.solve(A,b)
```

```{code-cell} ipython3
f1_n = np.insert(f1_n,0,y_0)
f1_n = np.append(f1_n,y_Nm1)
```

```{code-cell} ipython3
def f1_a(x):
    return -1.3*np.exp(-x)+2.9*np.exp(2*x)+np.exp(3*x)/4
```

```{code-cell} ipython3
xx1 = np.linspace(0,1,300)
plt.plot(xx1,f1_a(xx1),"b-")
plt.plot(x,f1_n,"r:")
plt.legend(["Analytic","Finite Differences"])
plt.show()
```

```{code-cell} ipython3
len(f1_n)
```

```{code-cell} ipython3
RMS_Q2 = np.sqrt(((f1_n - f1_a(x))**2).sum() / N)
RMS_Q2
```

# 3

Solve the following ODE for the given conditions:

$$y' = y -x\\y(1)=4.71828,\quad x\in [1,2];\quad h\le0.01$$


**Bonus:**

Its analytical solution is: $y(x)=e^x+x+1$


Calculate the total RMS error defined as:

$$\sqrt{\frac{1}{N}\sum_i{(y_i-t_i)^2}}$$

where $y$ is the model estimate and $t$ is the analytical solution's value (_i.e._, true value).

+++

## Solution

### Euler

```{code-cell} ipython3
N = 105
x2 = np.linspace(1,2,N)
h = x2[1] - x2[0]
y2 = [4.71828]

h
```

```{code-cell} ipython3
for i in range(N-1):
    y_ip1 = y2[i] + (y2[i] - x2[i])*h
    y2.append(y_ip1)
```

```{code-cell} ipython3
xx2 = np.linspace(1,2,100)
plt.plot(xx2,np.exp(xx2)+xx2+1,"b-")
plt.plot(x2,y2,"r:")
plt.legend(["Analytic","Euler"])
plt.show()
```

```{code-cell} ipython3
RMS_Q3_Euler = np.sqrt(((y2 - (np.exp(x2)+x2+1))**2).sum() / N)
RMS_Q3_Euler
```

### RK4

```{code-cell} ipython3
N = 105
x2 = np.linspace(1,2,N)
h = x2[1] - x2[0]
y2 = [4.71828]

h
```

```{code-cell} ipython3
def f(t,y):
    return y-t
```

```{code-cell} ipython3
def y_t(t):
    return np.exp(t)+t+1
```

```{code-cell} ipython3
y = [4.71828]
t = np.linspace(1,2,N)
h = t[1]-t[0]
t,h

print("{:>2s}\t{:>8s}\t{:^8s}\t{:>5s}"\
      .format("t","y_KR4","y_true","Err%"))
for i in range(1,N):
    k1 = f(t[i-1],y[i-1])
    k2 = f(t[i-1]+0.5*h,y[i-1]+0.5*k1*h)
    k3 = f(t[i-1]+0.5*h,y[i-1]+0.5*k2*h)
    k4 = f(t[i-1]+h,y[i-1]+k3*h)
    y.append(y[i-1]+(k1+2*k2+2*k3+k4)*h/6)

#print(len(y))
    
for i in range(len(y)):
    print("{:>2d}\t{:8.5f}\t{:8.5f}\t{:5.4f}%"\
      .format(i,y[i],y_t(t[i]),np.abs(y_t(t[i])-y[i])/y_t(t[i])*100))
```

```{code-cell} ipython3
xx2 = np.linspace(1,2,100)
plt.plot(xx2,np.exp(xx2)+xx2+1,"b-")
plt.plot(t,y,"r:")
plt.legend(["Analytic","RK4"])
plt.show()
```

```{code-cell} ipython3
RMS_Q3_RK4 = np.sqrt(((y - (np.exp(t)+t+1))**2).sum() / N)
RMS_Q3_RK4
```

```{code-cell} ipython3

```
