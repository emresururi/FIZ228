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

# Resit Exam Solutions (20232)
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

28/06/2024

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
```

**Pick and solve any 3 of the following questions**

_(No bonus will be given for a 4th one!)_

+++

# 1

a. Construct a pandas dataframe with 5 (imaginary) students’ information containing their:    
   <pre>
        1. Name,  
        2. Surname,  
        3. ID #,  
        4. FIZ227 - Programming Letter Grade  
        5. FIZ228 - (Prospective) Final Exam Grade  
        6. Overall Grade Average  </pre>
b. Add your information as the 6th entry  
c. Calculate the average of the "FIZ228 - (Prospective) Final Exam Grade" column and round it  
d. Have the pandas return the information of the student with the highest "Overall Grade Average"

+++

## Solution

### a

```{code-cell} ipython3
data = {'Name':["Ayşe","Banu","Ceren","Defne","Elif"],
        'Surname':["Ahmet","Barış","Cengiz","Durmuş","Erkin"],
        'ID':[123,234,345,456,567],
        'FIZ227':["A1","B1","C1","D","F3"],
        'FIZ228':[80,82,76,65,42],
        'Overall Average':[3.2,2.6,3.7,2.8,1.7]
       }
df = pd.DataFrame(data)
df
```

### b

```{code-cell} ipython3
df.loc[5] = ["Emre","Taşcı",999,"A3",86,3.5]
df
```

### c

```{code-cell} ipython3
np.round(df["FIZ228"].mean())
```

### d

```{code-cell} ipython3
id_max = df["Overall Average"].idxmax()
df.loc[id_max]
```

# 2 

On a 1D wire, some measurements have been made and the following data have been obtained:

|x|y|
|---|---|
|-1|0|
|2|-24|
|4|0|
|6|0|

In the table above, $x$ denotes the distance from a point marked on the wire while $y$ corresponds to the magnitude of the property measured.

Given that the measured property is a continous quantity, estimate:

a) The value at the x = 5 position.  
b) The position where the measured property's value is equal to -75 (within a precision of the order 10<sup>-4</sup>).

_(For the sake of simplicity, units have been ignored.)_

+++

## Solution

Since there are 3 roots (-1,2,4), we know that the simplest fit will be through a 3rd order polynomial.

```{code-cell} ipython3
n = 3
xs = np.array([-1,2,4,6])
ys = np.array([0,-24,0,0])
qq = np.polyfit(xs,ys,n)
qq
```

```{code-cell} ipython3
poly = np.poly1d(qq)
print(poly)
```

```{code-cell} ipython3
x = np.linspace(-4,10,300)
plt.plot(x,np.polyval(qq,x),"r-")
plt.plot(xs,ys,"ko")
plt.show()
```

### a

```{code-cell} ipython3
np.polyval(qq,5)
```

### b

If $x_0$ is the sought value that corresponds to $y(x_0) = -75$, then the (current polynomial - (-75)) will have a root at that position:

```{code-cell} ipython3
poly75 = poly - (-75)
print(poly75)
```

```{code-cell} ipython3
roots75 = poly75.roots
roots75
```

We pick the only real value:

```{code-cell} ipython3
x0 = roots75[0].real
x0
```

```{code-cell} ipython3
poly(x0)
```

**Alternative: Brute-force solution**

Checking the plot above, we see that it takes -75 value in the region $x\in[6,9]$, so start from left and go towards right, until we are within a $\Delta = 10^{-4}$ neigbourhood. Applying a simple bisection method, we can reach there:

```{code-cell} ipython3
x_a = 6
p_a = poly(x_a)

x_b = 9
p_a = poly(x_b)

x_c = (x_a+x_b)/2
p_c = poly(x_c)

target_y = -75

diff = np.abs(poly(x_c) - target_y)
while(diff>1E-4):
    if((p_a>target_y) & (p_c<target_y)):
        # The sought x0 is between a & c
        # a<-a, b<-c
        x_b = x_c
        p_b = p_c
    else:
        # The sought x0 is between c & b
        # a<-c, b<-b
        x_a = x_c
        p_a = p_c
    
    # update the c to be the midpoint of (a,b)
    x_c = (x_a+x_b)/2
    p_c = poly(x_c)
    
    diff = np.abs(poly(x_c) - target_y)

print(x_c,poly(x_c))
              
            
```

# 3

Suppose that we have two kinds of particles: A and B (you can think of them as golf balls and tennis balls). When they are put in a system, the system's energy is given by the following formula:

$$U(n_A,n_B) = -2n_A^2-n_B^2+5n_An_B+10n_A+70n_B$$

where $n_{\{A,B\}}$ indicates the number of A and B particles in the system.

Due to the system's capacity, and the fact that B particles being bigger than A particles, we have the following restriction:

$$n_A + 2.7n_B \le 101$$

Find the optimal number of A and B particles to be put into the system such that they satisfy the above restriction while yielding the maximum energy.

+++

## Solution

+++

Since we are after the _maximum_ of the function, we _minimize_ the negative of it:

```{code-cell} ipython3
def f(xy):
    x = xy[0]
    y = xy[1]
    return -(10*x+70*y-2*x**2-y**2+5*x*y)
```

```{code-cell} ipython3
constraint = opt.LinearConstraint([1,2.7],0,100.999)
```

```{code-cell} ipython3
res = opt.minimize(f,[19,30],constraints=constraint,bounds=[(0,100),(0,None)])
res
```

As we are dealing with _the number of the balls_, i.e., integers, rounding them gives us the solution:

```{code-cell} ipython3
n_A = (int)(np.round(res.x[0]))
n_B = (int)(np.round(res.x[1]))
n_A,n_B
```

Checking if they satisfy the constraint:

```{code-cell} ipython3
n_A + 2.7*n_B < 101
```

and this configuration's energy:

```{code-cell} ipython3
-f([n_A,n_B])
```

**Alternative: Brute-force solution**

As in this problem there are not so many possible options ($0\le n_A\lt101$ and $0\le n_B\lt \frac{101}/{2.7}\approx 37$), we can go over all the possibilities:

```{code-cell} ipython3
n_A_max = 0
n_B_max = 0
f_max = 0

for n_A in range(101):
    for n_B in range(37):
        if(n_A + 2.7*n_B >= 101):
            continue
        # Don't forget that the defined function 
        # is yielding the negative of the given
        # one due to the "minimization"->"maximization"
        ff = -f((n_A,n_B))
        if(ff>f_max):
            n_A_max = n_A
            n_B_max = n_B
            f_max = ff

print(n_A_max,n_B_max,-f((n_A_max,n_B_max)))
        
```

# 4

Solve the following ODE for the given conditions:

$$y'' + y - e^{-x/10} y' = 0\\y(0)=1,\,y'(0)=0\quad x\in [0,12];\quad h\le0.01$$

Plot your result.

+++

## Solution

```{code-cell} ipython3
def f(x,y,yp):
    # y'' = f(x,y,yp)
    return -y+np.exp(-x/10)*yp
```

### Euler

```{code-cell} ipython3
x_E = np.linspace(0,12,20000)
h = x_E[1] - x_E[0]
print(h)

y_E = np.array([1])
yp_E = np.array([0])

for xx in x_E[:-1]:
    yp_ip1_E = yp_E[-1] + f(x_E[-1],y_E[-1],yp_E[-1]) *h
    yp_E = np.append(yp_E,yp_ip1_E)
    y_ip1_E = y_E[-1] + yp_ip1_E * h
    y_E = np.append(y_E,y_ip1_E)
```

```{code-cell} ipython3
plt.plot(x_E,y_E,"-")
plt.show()
```

### RK4

```{code-cell} ipython3
x_RK = np.linspace(0,12,20000)
h = x_RK[1] - x_RK[0]
print(h)

y_RK = np.array([1])
yp_RK = np.array([0])

for i in range(x_RK.size-1):
    k1 = f(x_RK[i],y_RK[-1],yp_RK[-1])
    k2 = f(x_RK[i]+0.5*h,y_RK[-1],yp_RK[-1]+0.5*k1*h)
    k3 = f(x_RK[i]+0.5*h,y_RK[-1],yp_RK[-1]+0.5*k2*h)
    k4 = f(x_RK[i]+h,y_RK[-1],yp_RK[-1]+k3*h)
    yp_ip1_RK = yp_RK[-1]+(k1+2*k2+2*k3+k4)*h/6
    yp_RK = np.append(yp_RK,yp_ip1_RK)

    y_ip1_RK = y_RK[-1] + yp_ip1_RK * h
    y_RK = np.append(y_RK,y_ip1_RK)
```

```{code-cell} ipython3
plt.plot(x_RK,y_RK,"-b")
plt.show()
```

### Built-in (RK45)

_This is included for reference purposes only, as it is a little bit advanced!_

```{code-cell} ipython3
from scipy import integrate
```

```{code-cell} ipython3
def ODEs(x,y):
    # here y is an array holding y and y'
    # so first, we derive y'
    yp = y[1] # y'
    # and use it to derive y''
    ypp = -y[0]+np.exp(-x/10)*yp # y''
    return [yp,ypp]
```

```{code-cell} ipython3
RK45_solution = integrate.solve_ivp(ODEs,[0,12],[1,0],
                                    t_eval=np.linspace(0,12,500))
RK45_solution
```

```{code-cell} ipython3
x_RK45 = RK45_solution.t
y_RK45 = RK45_solution.y[0]
plt.plot(x_RK45,y_RK45,"-")
plt.show()
```

### Comparison

```{code-cell} ipython3
plt.plot(x_E,y_E,"-r")
plt.plot(x_RK,y_RK,"-b")
plt.plot(x_RK45,y_RK45,"--g")
plt.legend(["Euler","RK4 (home made)","RK45 (built-in)"])

plt.show()
```

This comparison certainly proves the superiority of the Runge-Kutta over Euler! Even though Euler captures the behavior, the scale is not correct.

```{code-cell} ipython3

```
