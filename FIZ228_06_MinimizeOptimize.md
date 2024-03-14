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

# Minimization & Optimization
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

Although we have already studied and employed various minimization commands and used them in conjunction within the optimization problems (by minimizing the errors to fit given models), a deeper insight and variations might prove useful.

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
np.random.seed(228)
```

## scipy.optimize.minimize[*](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

`minimize()` function from scipy's _optimize_ module handles a variety of minimization methods (if not specifically denoted, then by default uses the [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) method (for unconstrained problems)).

+++

### Single variable function
Consider a "negative gaussian" function, so instead of a peak, it will have a bottom, _i.e.,_ a definite minimum that we are seeking. We know that it will be located at the $\mu$. **However, don't forget that, in real-life situations we have no idea what the function is let alone where its minimum is.**

Pay attention that the variable $x$ is defined first and the parameters $(\mu,\sigma)$ come afterwards in the function declaration.

```{code-cell} ipython3
def f(x,mu,sigma):
    return -np.exp(-(x-mu)**2/(2*sigma)**2)
```

```{code-cell} ipython3
mu = 5
sigma = 0.7
x = np.linspace(0,10,300)
plt.plot(x,f(x,mu,sigma),"k-")
plt.show()
```

When calling the `minimize` function, we feed an initial guess / starting point for the x variable (here: 3) and supply the values of the characteristic parameters $(\mu,\sigma)$. 

_Once again: we are searching for the minimum -- not looking for the optimal parameters that fits a function to a given set of data!_

```{code-cell} ipython3
# mu and sigma here have definite values (5 & 0.7)
opt.minimize(f,3,(mu,sigma))
```

### Multi-variate function

As we have observed from the single variable function example, `minimize` takes the first parameter as the variable. Thus, when we are dealing with a multi-variate function, we need to implement it as a vector variable when defining the function.

Therefore a sample function of:

$$f(x,y;a,b) = (x-a)^2 + (y-b)^2$$

is defined as follows (a variable, followed by the parameters):

```{code-cell} ipython3
def f(xy,a,b):
    return (xy[0]-a)**2+(xy[1]-b)**2
```

Let's place the minimum at $(x_0,y_0) = (3,4)$:

```{code-cell} ipython3
x = np.linspace(-1,7,100)
y = np.linspace(-1,9,100)
(xx,yy) = np.meshgrid(x,y)
zz = f((xx,yy),3,4)
```

```{code-cell} ipython3
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

plt.xlabel('x')
plt.ylabel('y')

surf = ax.plot_surface(xx, yy, zz, cmap = plt.cm.viridis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()
```

```{code-cell} ipython3
#levels = np.arange(0,15,3)
levels = np.array([0,0.1,0.2,0.5,0.7,1,2,3,5,8,12,15])

lo1 = plt.contourf(xx, yy, zz, cmap = plt.cm.summer,levels=levels)
plt.grid(False)
plt.rcParams['axes.grid'] = False
plt.colorbar()

lo2 = plt.contour(xx, yy, zz, colors='k',levels=levels)
plt.xlabel('x')
plt.ylabel('y')
plt.clabel(lo2, inline=True, fontsize=10)


plt.grid()

plt.show()
```

And here is how we find its minimum (by starting from (x<sub>0</sub>,y<sub>0</sub>) = (1,2) and characterizing the function by setting the (a,b) parameters to (3,4)):

```{code-cell} ipython3
opt.minimize(f,[1,2],(3,4))
```

### Minimization with constraints

Sometimes, one or more constraints is present. Suppose that for our previously defined function $f(x,y;a,b) = (x-a)^2 + (y-b)^2$, we want to minimize it but we also want to satisfy the condition of $2x + 3y = 5$. This is where the `constraints` parameter comes into play: the constraint (in our case, we'll be dealing only with linear constraints such as $a_0x_0 + a_1x_1 + \dots + a_nx_n = E$ -- we can even flex this constraint into a more generalized inequality, such as:

$$E_{min} \le a_0x_0 + a_1x_1 + \dots + a_nx_n \le E_{max}$$

or more formally as a cofficient matrix - vector multiplication:

$$E_{min} \le \begin{pmatrix}a_0&a_1&\dots&a_n\end{pmatrix}\cdot\begin{pmatrix}x_0\\x_1\\\vdots\\ x_n\end{pmatrix}\le E_{max}$$

Returning to our example, we can write our constraint $2x + 3y = 5$ as a coefficient matrix - vector multiplication, bounded from bottom and top in the form of:

$$5 \le \begin{pmatrix}2&3\end{pmatrix}\cdot\begin{pmatrix}x\\y\end{pmatrix}\le 5$$

_See that, by setting the lower and upper bound to the same value, we have thus defined an equality._

```{code-cell} ipython3
# This is our coefficient matrix
A = np.array([2,3])
```

We introduce our constraint via the `opt.LinearConstraint()` method, where the first parameter is the coefficient matrix, followed by the lower and upper bounds, respectively:

```{code-cell} ipython3
opt.minimize(f,[1,2],(3,4),constraints=opt.LinearConstraint(A,5,5))
```

**Example: Heron's Formula**

Heron's formula (formulated by [Heron of Alexandria](https://en.wikipedia.org/wiki/Heron_of_Alexandria), 1st century) is used to calculate the area of a triangle when its side lengths $(a,b,c)$ are known. 

If $s$ is defined as $s = \tfrac{1}{2}(a+b+c)$, then the area $A$ is given by:

$$A=\sqrt{s(s-a)(s-b)(s-c)}$$

or, alternatively as:

$$A=\tfrac{1}{4}\sqrt{4a^2b^2-(a^2+b^2-c^2)^2}$$

```{code-cell} ipython3
def HeronTri(x):
    # Heron's Formula
    return np.sqrt(4*x[0]**2*x[1]**2 - (x[0]**2+x[1]**2-x[2]**2)**2)/4
```

Suppose that we have a rope of length 18 cm, and we want to form a triangle with the lowest area, given that its side lengths are at least 2 cm.

_Why at least 2 cm?_

Otherwise it would be a boring question as we could take one side to be almost 0 and divide the remaining length to 2:

```{code-cell} ipython3
abc = (1E-8,(18-1E-8)/2,(18-1E-8)/2)
abc,HeronTri(abc),np.sum(abc)
```

Here are two possible cases that come to mind and their corresponding areas:

```{code-cell} ipython3
HeronTri([6,6,6])
```

```{code-cell} ipython3
HeronTri([5,6,7])
```

Can we do better?

+++

Since we are dealing with a triangle, the triangle inequalities must also be obeyed, _i.e.,_:

* $ a + b > c \rightarrow a + b - c > 0$
* $ b + c > a \rightarrow -a + b + c > 0$
* $ c + a > b \rightarrow a - b + c > 0$

along with the constraint due to the rope-length: $a+b+c = 18$

Combining all, we have the following constraints:

```{code-cell} ipython3
con_a = opt.LinearConstraint([1,0,0],2,18) # 2 <= a <= 18
con_b = opt.LinearConstraint([0,1,0],2,18) # 2 <= b <= 18
con_c = opt.LinearConstraint([0,0,1],2,18) # 2 <= c <= 18
con_d = opt.LinearConstraint([1,1,-1],1E-3,18) # 0 <  a + b - c <= 18
con_e = opt.LinearConstraint([-1,1,1],1E-3,18) # 0 < -a + b + c <= 18
con_f = opt.LinearConstraint([1,-1,1],1E-3,18) # 0 <  a - b - c <= 18
con_g = opt.LinearConstraint([1,1,1],18,18) # a + b + c = 18
cons = [con_a,con_b,con_c,con_d,con_e,con_f,con_g]
```

(Instead of specifying each constraint individually, we could have collected them in a matrix -- see the "[Collecting constraints](#collecting_constraints)" heading below)

We have set the upper limits of the sides and the inequalities to the rope length because it makes sense! ;)

```{code-cell} ipython3
res = opt.minimize(HeronTri,[3,4,5],constraints=cons)
res
```

```{code-cell} ipython3
HeronTri(res.x),res.x.sum()
```

Even though it's a very boring triangle, it satisfies all the conditions, including the triangle inequalities:

```{code-cell} ipython3
(x,y,z) = res.x
```

```{code-cell} ipython3
x + y > z
```

```{code-cell} ipython3
y + z > x
```

```{code-cell} ipython3
x + z > y
```

**Challenge #1:** Can you find the triangle with the minimum area subject to to the above constraints, but also satisfies the condition such that the difference between any two sides is less than 3?

+++

**Challenge #2:** What about the maximum area yielding triangle, subject to the condition that the sum of its side lengths is equal to 18?

+++

**Side information:**

Analytically, these kind of minimization problems with constraints are usually solved using a wonderful technique called _Lagrange Multipliers<sup>[1](https://tutorial.math.lamar.edu/classes/calciii/lagrangemultipliers.aspx),</sup><sup>[2](https://math.libretexts.org/Bookshelves/Calculus/Calculus_\(OpenStax\)/14%3A_Differentiation_of_Functions_of_Several_Variables/14.08%3A_Lagrange_Multipliers)</sup>_.

+++

**Lesson to learn: Starting values** 

If we had chosen a different starting point than (3,4,5) in our search above, we'd -most likely- still be able to land at the same minimum, e.g.,

```{code-cell} ipython3
res = opt.minimize(HeronTri,[3,8,10],constraints=cons)
res
```

```{code-cell} ipython3
res = opt.minimize(HeronTri,[7,11,5],constraints=cons)
res
```

However, check what happens when we introduce symmetry:

```{code-cell} ipython3
res = opt.minimize(HeronTri,[3,3,3],constraints=cons)
res
```

Taking a starting value of $a=b=c$ effects the algorithm such that, due to the symmetry, it opts to move in the same direction, hence, ending at the worst solution. A similar issue ensues for lower symmetries ($a=b\ne c;a\ne b=c;a=c\ne b$) as well:

```{code-cell} ipython3
res = opt.minimize(HeronTri,[3,3,5],constraints=cons)
res
```

Thus, always make sure that you chose various starting points and don't incorporate symmetries unless there's a specific reason to do so!

+++

**Collecting constraints**<a id='collecting_constraints'></a>

In the above example, we defined each constraint separately and then collected them in an array but remembering that $A$ is the coefficient matrix, we could have all collected them in $A$:

```{code-cell} ipython3
A=np.array([[1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,-1],
            [1,-1,1],
            [-1,1,1]])
con_inequalities = opt.LinearConstraint(A,[2,2,2,1E-3,1E-3,1E-3],18)
con_equality = opt.LinearConstraint([1,1,1],18,18)
cons = [con_inequalities,con_equality]

res = opt.minimize(HeronTri,[3,4,5],constraints=cons)
res
```

A couple of things to consider:

* We grouped the inequality constraints together but left the equality constraint out -- this is for efficiency (if we had also included it, then Python would have warned us to keep them separate)

* We have individually declared the lowerbounds whereas for the upperbound, we entered a single value as it is common (if we wanted, we could have of course, typed 18 six times as an array)

+++

# Gradient Descent Algorithm

In order to grasp the meaning of the minimization procedure, we will explicitly investigate the gradient descent algorithm.

This algorithm works by moving along the steepest direction and steeper the slope, the bigger steps we take.

Imagine that you find yourself on the side of a hill, surrounded by a mist, unable to see 1 meter ahead and you want to reach to the bottom of the hill (and for some reason, you can't walk but teleport (meaning that, you don't interact with whatever is between you and your destination -- can't see, can't feel)). What you do is, you move along the down slope. Mathematically speaking, for a 1-dimensional graph of a function, if the slope at the place you are positioned is positive (positive slope means that as x increases, the function value increases), it means that the down is towards left (i.e., if you pour some water to the ground, it will flow downwards: so, <u>slope positive -> move towards left (-x direction)</u> and vice versa (<u>slope negative -> move towards right</u>)). Since the slope of a function at any point is defined by its derivative, it's very helpful to have the function's derivative also defined, if possible.

As it's much easier to understand the procedure by working on an example, we'll cover a parabola and the "negative gaussian" function we have already defined at the beginning of this lecture.

+++

## Parabola

```{code-cell} ipython3
def f(x,abc):
    return abc[0]*x**2+abc[1]*x+abc[2]

def g(x,abc):
    # Derivative
    return 2*abc[0]*x+abc[1]
```

```{code-cell} ipython3
xx = np.linspace(-5,10,100)
abc = np.array([2,3,-4])
plt.plot(xx,f(xx,abc))
plt.show()
```

So, this is the function we are trying to find its minimum. Once again, don't forget that, we do not necessarily know the function or what it looks like: it might be a property of a material/medium like the location of a room with the minimal electrical field, the concentration of a component in a compound that yields the most optimal conductivity, or it can be a function so complicated that even though we want to calculate its minimum, can't calculate its derivative to set it to zero and solve that way analytically...

It is assumed that we ask ("shoot") for the value at the position we are interested, and for each of our queries, we have to spend some time/money/material, so the sooner we reach the minimum the better!

We have to start from somewhere... Let's start from x = 5: at this point, the slope is:

```{code-cell} ipython3
g(5,abc)
```

It is positive (we will walk towards left) and has a value of 23 (we will take a big step). The slope's value at that point determine the step size since as we move closer to the minimum, the slope magnitude decreases:

```{code-cell} ipython3
plt.plot(xx,f(xx,abc))
i = 7
for x in np.arange(9.75,-0.76,-1.75):
    x12 = np.array([x-0.35,x+0.35])
    plt.plot(x12,f(x12,abc),"r-")
    plt.text(x,f(x,abc)-20,str(x)+": "+str(g(x,abc)))
    i -= 1
    print("x: {:5.2f} | Slope: {:5.2f}".format(x,g(x,abc)))
plt.axis([-5,13,-50,250])
plt.show()
```

As you can check from the red lines depicting the slopes, their value decrease as one moves towards the minimum.

Only at the minimum(/maximum) the slope becomes 0 as it is the characteristic of extrema. Since we are moving along the decrease direction of the slope, we are moving towards the minimum.

Depending on the function, taking a step size equal to the slope value at that position might prove to be too big or too little. We compensate this fact by using a factor usually labelled as $\eta$. $\eta$ isn't necessarily taken as constant, it can be coded to be updated as the slope values process (adaptive approach).

Also, we can decide when to stop by defining a threshold tolerance for the magnitude of slope.

Let's apply these procedures to the parabola, step by step (keep in mind that we know the values of the function only at the specified positions):

```{code-cell} ipython3
:tags: [output_scroll]

x = 5
N = 50
eta = .4
tolerance = 1E-4
xs_so_far = [x]
fs_so_far = [f(x,abc)]
for i in range(N):
    gg = g(x,abc)
    print("Step #{:d}".format(i+1))
    print("The derivative (gradient) at x = {:7.5f} is {:7.5f}"\
          .format(x,gg))
    if(np.abs(gg)<tolerance):
        print("\tAs it is sufficiently close to zero, we have found the minima!")
        break
    elif(gg>0):
        print("\tAs it is positive, go left by: "+
              "(this amount)*eta(={:.2f}).".format(eta))
    else:
        print("\tAs it is negative, go right by: "+
              "|this amount|*eta(={:.2f}).".format(eta))

    delta = -gg*eta
    x0 = x
    x = x + delta
    xs_so_far.append(x)
    fs_so_far.append(f(x,abc))
    print("\t==> The new x is {:7.5f}{:+7.5f}={:7.5f}".format(x0,delta,x))
    plt.plot(xx,f(xx,abc),color="orange")
    plt.plot(xs_so_far,fs_so_far,"*-")
    plt.show()

    print("-"*45)
```

```{code-cell} ipython3
# Real minimum:# Real minimum:
np.roots([2*abc[0],abc[1]]) # root of 2ax + b
```

## "Negative" Gaussian

```{code-cell} ipython3
def f(x,mu,sigma):
    return -np.exp(-(x-mu)**2/(2*sigma**2))

def g(x,mu,sigma):
    return (x-mu)/(sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
```

```{code-cell} ipython3
mu = 5
sigma = 1
```

```{code-cell} ipython3
xx = np.linspace(0,15,100)
plt.plot(xx,f(xx,mu,sigma))
plt.show()
```

```{code-cell} ipython3
:tags: [output_scroll]

x = 8
N = 60
eta = .4
tolerance = 1E-4
xs_so_far = [x]
fs_so_far = [f(x,mu,sigma)]
for i in range(N):
    gg = g(x,mu,sigma)
    print("Step #{:d}".format(i+1))
    print("The derivative (gradient) at x = {:7.5f} is {:5.4f}"\
          .format(x,gg))
    if(np.abs(gg)<tolerance):
        print("\tAs it is sufficiently close to zero, we have found the minima!")
        break
    elif(gg>0):
        print("\tAs it is positive, go left by: "+
              "(this amount)*eta(={:.2f}).".format(eta))
    else:
        print("\tAs it is negative, go right by: "+
              "|this amount|*eta(={:.2f}).".format(eta))

    delta = -gg*eta
    x0 = x
    x = x + delta
    xs_so_far.append(x)
    fs_so_far.append(f(x,mu,sigma))
    print("\t==> The new x is {:7.5f}{:+7.5f}={:7.5f}".format(x0,delta,x))
    plt.plot(xx,f(xx,mu,sigma),color="orange")
    plt.plot(xs_so_far,fs_so_far,"*-")
    plt.show()

    print("-"*45)
```

# Fitting parameters via the gradient descent algorithm

```{code-cell} ipython3
mu = 5
sigma = 1

N = 10
x = np.random.rand(N)*4+3
t = f(x,mu,sigma)

xx = np.linspace(3,7,100)

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(x,t,"o")
plt.show()
```

$\newcommand{\diff}{\text{d}}
\newcommand{\dydx}[2]{\frac{\text{d}#1}{\text{d}#2}}
\newcommand{\ddydx}[2]{\frac{\text{d}^2#1}{\text{d}#2^2}}
\newcommand{\pypx}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\unit}[1]{\,\text{#1}}$

We have the data points, we know the function but we don't have the mu & sigma.

$$f(x;\mu,\sigma)=-\exp{\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]}$$

The function we are going to try to minimize will be the difference between the real values ($\{t_i\}$) corresponding to $\{x_i\}$ and the projected values ($\{y_i\}$):

$$F(x_i,t_i,\mu,\sigma) = t_i - f(x_i;\mu,\sigma)$$

Begin by calculating the derivatives:

$$\pypx{F}{\mu}=\frac{x_i-\mu}{\sigma^2}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\\
\pypx{F}{\sigma}=\frac{(x_i-\mu)^2}{\sigma^3}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}$$

_(don't forget that $\{x_i\}$ and $\{t_i\}$ are fixed!)_

Can you see the problem in this approach? As $\{t_i\}$ are fixed, the problem is reduced to finding the $(\mu,\sigma)$ set that will make $f(x_i;\mu,\sigma)$ minimum, regardless of $\{t_i\}$ values. If we follow this approach, we will end up with $(\mu,\sigma)$ that will most likely fix the values all very close to 0.

You are invited to try this approach, i.e.,

```python
def F_mu(x,mu,sigma):
    return (x-mu)/sigma**2*np.exp(-(x-mu)**2/(2*sigma**2))
def F_sigma(x,mu,sigma):
    return (x-mu)**2/sigma**3*np.exp(-(x-mu)**2/(2*sigma**2))
```

But what we really have in mind is the fact that, for a given $x_i$, we want to find values as close to the corresponding $t_i$ as possible. One way to obtain this would be to define the error function as:

$$F(x_i,t_i,\mu,\sigma) = \left[t_i - f(x_i;\mu,\sigma)\right]^2=\left\lbrace t_i -
\left[-\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}\right]
\right\rbrace^2$$

and then we would have the following derivatives:

$$\pypx{F}{\mu} = \frac{2(x_i-\mu)}{\sigma^2}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\left\lbrace t_i+\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\right\rbrace\\
\pypx{F}{\sigma}=\frac{2(x_i-\mu)^2}{\sigma^3}\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\left\lbrace t_i+\exp{\left[-\frac{(x_i-\mu)^2}{2\sigma^2}\right]}\right\rbrace$$

_(Evaluated via WolframAlpha: [1](https://www.wolframalpha.com/input?i=d%2Fdu+%28t%2Bexp%28-%28x-u%29%5E2%2F%282*s%5E2%29%29%29%5E2), [2](https://www.wolframalpha.com/input?i=d%2Fds+%28t%2Bexp%28-%28x-u%29%5E2%2F%282*s%5E2%29%29%29%5E2))_

```{code-cell} ipython3
def F_mu(x,t,mu,sigma):
    return 2*(x-mu)/sigma**2*np.exp(-(x-mu)**2/(2*sigma**2))*\
(t+np.exp(-(x-mu)**2/(2*sigma**2)))

def F_sigma(x,t,mu,sigma):
    return 2*(x-mu)**2/sigma**3*np.exp(-(x-mu)**2/(2*sigma**2))*\
(t+np.exp(-(x-mu)**2/(2*sigma**2)))
```

```{code-cell} ipython3
np.array([x,t]).T
```

```{code-cell} ipython3
eta = 1

# Starting values
mu_opt = 2.7
sigma_opt = 2.3
tolerance = 1E-4

for i in range(10000):
    for ii in range(x.size):
        xi = x[ii]
        ti = t[ii]
        #print(xi,ti)
        F_mu_xi = F_mu(xi,ti,mu_opt,sigma_opt)
        F_sigma_xi = F_sigma(xi,ti,mu_opt,sigma_opt)
        mu_opt -= eta*F_mu_xi
        sigma_opt -= eta*F_sigma_xi
    total_absolute_error = np.sum(np.abs(t-f(x,mu_opt,sigma_opt)))
    if(total_absolute_error < tolerance):
        print(("As the sum of the absolute errors is sufficiently close to zero ({:.7f}),\n"+
              "\tbreaking the iteration at the {:d}. step!").
              format(total_absolute_error,i+1))
        break
print("mu: {:.4f}\tsigma: {:.4f}".format(mu_opt,sigma_opt))

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(xx,f(xx,mu_opt,sigma_opt),":b")
plt.plot(x,t,"o")
plt.show()
```

```{code-cell} ipython3
def f(x,mu,sigma):
    return -np.exp(-(x-mu)**2/(2*sigma**2))
```

```{code-cell} ipython3
# Doing the same thing via curve_fit():
# Unbounded
res,_ = opt.curve_fit(f,x,t,[2.7,2.3])
res
```

```{code-cell} ipython3
# Bounded
res,_ = opt.curve_fit(f,x,t,[2.7,2.3],bounds=[(2,0),(7,5)])
res
```

```{code-cell} ipython3
# And via optimize.minimize():
def F(musigma,x,t):
    return np.sum((t + np.exp(-(x-musigma[0])**2/(2*musigma[1]**2)))**2)
```

```{code-cell} ipython3
# Unbounded
res = opt.minimize(F,x0=(2.7,2.3),args=(x,t))
res.x,res.fun
```

```{code-cell} ipython3
# Bounded
res = opt.minimize(F,x0=(2.7,2.3),args=(x,t),bounds=[(3,6.5),(None,None)])
res.x,res.fun
```

## Stochastic Gradient Descent Algorithm (Optional)

In this approach, instead of optimizing the variables at every step for one data point, we optimize them as a whole:

```{code-cell} ipython3
eta = 0.1

# Starting values
mu_opt = 3.0
sigma_opt = 2.3
tolerance = 1E-4
total_absolute_error0 = 1000

for i in range(10000):
    d_mu    = -eta*np.sum(F_mu(x,t,mu_opt,sigma_opt))
    d_sigma = -eta*np.sum(F_sigma(x,t,mu_opt,sigma_opt))

    mu_opt    += d_mu
    sigma_opt += d_sigma

    total_absolute_error = np.sum(np.abs(t-f(x,mu_opt,sigma_opt)))

    if(total_absolute_error < tolerance):
        print(("As the sum of the absolute errors is sufficiently close to zero ({:.7f}),\n"+
              "\tbreaking the iteration at the {:d}. step!").
              format(total_absolute_error,i+1))
        break
print("mu: {:.4f}\tsigma: {:.4f}".format(mu_opt,sigma_opt))

plt.plot(xx,f(xx,mu,sigma),color="orange")
plt.plot(xx,f(xx,mu_opt,sigma_opt),":b")
plt.plot(x,t,"o")
plt.show()
```

# Case Study: 2 Springs, 1 Mass, 1 Side

+++

Consider the system that consists of a mass attached to springs from one side as shown in the figure:

![imgs/06_2Springs1Mass.png](imgs/06_2Springs1Mass.png)

The potential energy of the system is given by the equation:

$$ V(x) = \frac{1}{2}k_1(x-x_1)^2 + \frac{1}{2}k_2(x-x_2)^2$$

where $k_1$, $k_2$, $x_1$, and $x_2$ are constants and they are the spring constants and equilibrium lengths of the springs, respectively.

a) Find the equilibrium positions of the particle, i.e., the locations where the potential energy is minimum for:

$k_1 = 1 \text{ N/m},\, k_2 = 2 \text{ N/m},\, x_1= 0.3 \text{ m},\, x_2=0.7 \text{ m}$

```{code-cell} ipython3
def V(x,k1,k2,x1,x2):
    return 0.5*k1*(x-x1)**2 + 0.5*k2*(x-x2)**2
```

```{code-cell} ipython3
k1 = 1 # N/m
k2 = 2 # N/m
x1 = 0.3 # m
x2 = 0.7 # m

x = np.linspace(0,1,30)
Vx = V(x,k1,k2,x1,x2)

plt.plot(x,Vx,"k-")
plt.xlabel("x (m)")
plt.ylabel("V(x) (J)")
plt.show()
```

```{code-cell} ipython3
res = opt.minimize(V,0,(k1,k2,x1,x2))
res
```

```{code-cell} ipython3
V(17/30,k1,k2,x1,x2)
```

```{code-cell} ipython3
res = opt.root(V,0.6,(k1,k2,x1,x2))
res
```

```{code-cell} ipython3
V(res.x,k1,k2,x1,x2)
```

b) If its mechanical energy is given as 3 J, find the positions where its velocity is 0, i.e., _turning points_.

```{code-cell} ipython3
E = 3
def K(x,k1,k2,x1,x2):
    return E - V(x,k1,k2,x1,x2)
```

```{code-cell} ipython3
res = opt.root(K,x0=1,args=(k1,k2,x1,x2))
res 
```

```{code-cell} ipython3
K(res.x,k1,k2,x1,x2)
```

```{code-cell} ipython3
x = np.linspace(-1,3,30)
x_m = [-0.83491974,1.96825307]
plt.plot(x,K(x,k1,k2,x1,x2),"k-")
plt.plot(x_m,[0,0],"bo")
plt.plot(x,np.zeros(30),"--b")
Kx3 = K(3,k1,k2,x1,x2)
plt.plot([x_m[0],x_m[0]],[Kx3,0],":b")
plt.plot([x_m[1],x_m[1]],[Kx3,0],":b")
plt.plot([17/30,17/30],[Kx3,K(17/30,k1,k2,x1,x2)],":r")
plt.xlabel("x (m)")
plt.ylabel("K(x) (J)")
plt.show()
```

```{code-cell} ipython3
x = np.linspace(-1,3,30)
x_m = [-0.83491974,1.96825307]
plt.plot(x,K(x,k1,k2,x1,x2),"k-")
plt.plot(x,V(x,k1,k2,x1,x2),"r--")
plt.plot(x,E*np.ones(len(x)),":g")
plt.plot(x_m,[0,0],"bo")

plt.xlabel("x (m)")
plt.legend(["K(x)","V(x)","E"])
plt.show()
```

## Analytical Solutions

a)

$$V(x) = \frac{1}{2}k_1(x-x_1)^2 + \frac{1}{2}k_2(x-x_2)^2$$

$$\begin{align*} \rightarrow \dydx{V}{x}=k_1(x_0-x_1)+k_2(x_0-x_2) &= 0\\
 k_1x_0-k_1x_1+k_2x_0-k_2x_2 &= 0\\
 (k_1+k_2)x_0-(k_1x_1 + k_2x_2) &=0\\
 \Rightarrow x_0 = \frac{k_1x_1 + k_2x_2}{k_1+k_2}&
\end{align*}$$

```{code-cell} ipython3
(k1*x1+k2*x2)/(k1+k2)
```

b)

$$E - V(x_0) = K(x_0) = 0$$

$$\begin{align} V(x_0) &= E\\
 \frac{1}{2}k_1(x_0-x_1)^2 + \frac{1}{2}k_2(x_0-x_2)^2 &= E\\
 k_1(x_0-x_1)^2 + k_2(x_0-x_2)^2 &= 2E\\
 k_1x_0^2+k_1x_1^2 -2k_1x_1x_0+k_2x_0^2+k_2x_2^2 -2k_2x_2x_0&=2E\\
 \underbrace{(k_1+k_2)}_{a}x_0^2\underbrace{-2(k_1x_1+k_2x_2)}_{b}x_0+\underbrace{k_1x_1^2+k_2x_2^2-2E}_{c} &= 0\\
 \leftrightarrow ax_0^2+bx_0+c = 0
\end{align}$$

```{code-cell} ipython3
a = (k1+k2)
b = -2*(k1*x1+k2*x2)
c = (k1*x1**2+k2*x2**2-2*E)

delta = b**2-4*a*c
(-b+np.sqrt(delta))/(2*a),(-b-np.sqrt(delta))/(2*a)
```

```{code-cell} ipython3
np.roots([(k1+k2),-2*(k1*x1+k2*x2),(k1*x1**2+k2*x2**2-2*E)])
```
