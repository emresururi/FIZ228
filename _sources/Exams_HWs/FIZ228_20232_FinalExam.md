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

# 3

Solve the following ODE for the given conditions:

$$y' = y -x\\y(1)=4.71828,\quad x\in [1,2];\quad h\le0.01$$


**Bonus:**

Its analytical solution is: $y(x)=e^x+x+1$


Calculate the total RMS error defined as:

$$\sqrt{\frac{1}{N}\sum_i{(y_i-t_i)^2}}$$

where $y$ is the model estimate and $t$ is the analytical solution's value (_i.e._, true value).
