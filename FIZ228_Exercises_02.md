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

# Exercise Set #2
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

# Anscombe's Quartet

>Anscombe's quartet comprises four data sets that have nearly identical simple descriptive statistics, yet have very different distributions and appear very different when graphed. Each dataset consists of eleven (x, y) points. They were constructed in 1973 by the statistician Francis Anscombe to demonstrate both the importance of graphing data when analyzing it, and the effect of outliers and other influential observations on statistical properties. He described the article as being intended to counter the impression among statisticians that "numerical calculations are exact, but graphs are rough".[1]

[1] Anscombe, F. J. (1973). "Graphs in Statistical Analysis". American Statistician. 27 (1): 17–21. doi:10.1080/00031305.1973.10478966. JSTOR 2682899.

(From [Wikipedia entry](https://en.wikipedia.org/wiki/Anscombe%27s_quartet))

+++

|Ax|Ay|Bx|By|Cx|Cy|Dx|Dy|
|----|----|----|----|----|----|----|----|
| 10.0 |  8.04 | 10.0 |  9.14 | 10.0 |  7.46 |  8.0 |  6.58
|  8.0 |  6.95 |  8.0 |  8.14 |  8.0 |  6.77 |  8.0 |  5.76
| 13.0 |  7.58 | 13.0 |  8.74 | 13.0 | 12.74 |  8.0 |  7.71
|  9.0 |  8.81 |  9.0 |  8.77 |  9.0 |  7.11 |  8.0 |  8.84
| 11.0 |  8.33 | 11.0 |  9.26 | 11.0 |  7.81 |  8.0 |  8.47
| 14.0 |  9.96 | 14.0 |  8.10 | 14.0 |  8.84 |  8.0 |  7.04
|  6.0 |  7.24 |  6.0 |  6.13 |  6.0 |  6.08 |  8.0 |  5.25
|  4.0 |  4.26 |  4.0 |  3.10 |  4.0 |  5.39 | 19.0 | 12.50
| 12.0 | 10.84 | 12.0 |  9.13 | 12.0 |  8.15 |  8.0 |  5.56
|  7.0 |  4.82 |  7.0 |  7.26 |  7.0 |  6.42 |  8.0 |  7.91
|  5.0 |  5.68 |  5.0 |  4.74 |  5.0 |  5.73 |  8.0 |  6.89

_(You can download the data as CSV from the course page: Exercise_02_AnscombeQuartet.csv)_

+++

1. Import the data set as a pandas dataframe.
2. Plot each in separate graps.
3. Calculate the mean of each group's x and y values.
4. Calculate the variance of each group's x and y values.
5. Find the equation for the best fitting line for each group
6. Calculate the following quantities for each fit:
     * Sum of the squares of the data residuals ($S_t$)
     * Coefficient of variation (think about what $s_y$ in the formula represents ;)
     * Sum of the squares of the estimate residuals ($S_r$)
     * Standard error of the estimate ($s_{y/x}$)
     * Coefficient of determination ($r^2$)

+++

Interpret your findings. 

_For a more modern and interesting similar case, check [the Datasaurus Dozen](https://blog.revolutionanalytics.com/2017/05/the-datasaurus-dozen.html)! 8)_
