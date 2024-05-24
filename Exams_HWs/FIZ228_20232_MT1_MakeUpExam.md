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

# Midterm Exam #1 Make-Up (20232)
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

23/05/2024

+++

1. Import the data from the {download}`FIZ228_20232_MT1_MakeUp_data.csv<../data/FIZ228_20232_MT1_MakeUp_data.csv>` file.
2. Fit each one of the following function forms to the data:

$$f_1(x;A,\mu,\sigma,b) = A\exp[-\frac{(x-\mu)^2}{2\sigma^2}] + b$$

$$f_2(x;A,\omega,\phi,b) = A\sin(\omega x+\phi) + b$$

3. Calculate the error estimations between the given data and your model using the following error function:$\DeclareMathOperator\erf{erf}$

 * $\erf = \sqrt{\sum_{i}{(y_i - e_i)^2}}$
 
 where $y$ indicates the given data and $e$ the model estimation.
 
4. Calculate the coefficient of determination ($r^2$) for each model.

5. Which one of the functions would you pick for the most representative of the data? Briefly explain.

+++

_Submit your answer as a zip file containing both the html and the ipynb formats of your jupyter notebook, all the filenames formatted as "FIZ228_20232_MT1MU_NameSurname" (.html, .ipynb, .zip)_

```{code-cell} ipython3

```
