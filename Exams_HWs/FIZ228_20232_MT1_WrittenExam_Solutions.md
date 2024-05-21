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

# Midterm Exam #1 (20232) Written Session Solutions
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

19/04/2024

+++

Assume that you have been given a dataset with x and corresponding y values. 

You have two theoretical models for the related phenomenon, let’s say $f_1(x)$  and $f_2(x)$. These functions can be of very different varieties and can require completely different parameters (e.g., $f_1(x)$ can be a Gaussian like: $f_1(x;\mu,\sigma) = e^{-\frac{\left(x-\mu\right)^2}{2\sigma^2}}$, while $f_2(x)$ can be a sinusoidal function like: $f_2(x;A,\omega,\varphi) = A\sin(\omega x + \varphi)$).

Describe in detail, starting from the very beginning, how you would determine of the two proposed models, the one that fits the given data best. Please present your process, clearly and in steps, like phrasing a recipe:

1. _Do this by using ..._
2. _Calculate the … _
3. _…_

_You are not being asked to write a code (actually, please do not include any code!) – imagine transcribing this recipe to a fellow programmer working alongside: they know how to code but not what to do, so you need to tell them what is to be done, step by step. If you want them to calculate the error, you need to specify what you mean by “the error” (the formula); if you want them to fit the data, you need to specify what you mean by “fitting the data”, how it is done, on what criteria. (Any parts not covered by your instructions will be used against you!” ;)_

+++

## Solution

1. Define an error function – the many candidate error functions, let’s pick the root means squares, i.e. $\operatorname{errf}(y,e) = \sqrt{\sum_i{(y_i-e_i)^2}}$. (Here, $y$ indicates the given data values and $e$ indicates the estimated values obtained using the tried parameter set.)

2. Derive the optimized parameters by minimizing the error function. This will yield $\text{params}_1$ and $\text{params}_2$ parameter sets. Meaningful starting positions obtained by considering the given data and the candidate function's form will be most beneficial (e.g., if we are fitting a Gaussian, the position of the peak might be taken as $\mu_0$ and the spread value can be used to define $\sigma_0$; or if the fitting function is a sinusoidal, we can check the value range to come up with $A_0$ (and the shift along the y axis) and we can roughly estimate the period, hence deriving $\omega_0$, etc.). To further refine the process, the boundaries for the parameters can also be specified considering the data at hand.

3. Check if the parameters yield meaningful results (e.g., for a Gaussian function, $\sigma$ can not be negative; for a sinusoidal function, the frequency can not be negative; like [the case we did in the class](../FIZ228_04_Regression.html#really-which-one-is-better), if we know that the given data is $v$ vs. $F_{\text{drag}}$, the force can not take negative values for small values of $v$, etc.) 

4. Pick the function that has the minimum error given that its obtained parameters satisfy the 3rd criterion.
