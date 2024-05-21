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

# Midterm Exam #1 Hands On Session
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
Import the given {download}`FIZ228_20232_MT1_data.csv<../data/FIZ228_20232_MT1_data.csv>` file as a pandas dataframe.

This data is known to contain signals taken from two joint sinusoidal signal generators (where superposition applies, i.e., $y = y_1 + y_2$). 

Rename the first column as "t" and the second as "y"

+++

# 2

+++

Plot its graph as using red dots (no lines).

+++

# 3

As mentioned above, this data consists of the superposition of two sinusoidal signals, such that:

$$y = A_1\sin(\omega_1 t + \varphi_1)+A_2\sin(\omega_2 t + \varphi_2)$$

Determine the parameters (amplitude, angular frequency and phase) of these two sinusoidal signals that best match to the data given.

(Hint: Depending on your method, you can relax the default tolerance, in other words, don't get upset if a very high precision match can't be obtained)

+++

# 4

Using these parameters, construct your estimation function and introduce a third column labeled "e" on your dataframe that holds the function values with respect to your t values.

+++

# 5

Plot the given data against the model estimation (data points: red dots as above; model estimation: dashed line)

+++

# 6

Calculate the following error estimations between the given data and your model:$\DeclareMathOperator\erf{erf}$

 * $\erf_1 = \sqrt{\sum_{i}{(y_i - e_i)^2}}$
 * $\erf_2 = \sum_{i}{|y_i - e_i|}$
 
 where $y$ indicates the given data and $e$ the model estimation.
 
 Also calculate the coefficient of determination ($r^2$).

+++

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
