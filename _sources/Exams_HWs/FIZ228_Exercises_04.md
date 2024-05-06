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

# Exercise Set #4
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

## k-means Clustering

1. Import the data given in the {download}`Exercise_04_a.csv<../data/Exercise_04_a.csv>` file
2. Plot them.
3. Looking at your data, decide the optimal number of means ($k\in[2,6]$) and indicate them by drawing over by hand.
4. Device an algorithm to automatically come up with the optimal number of k-means and implement it.

+++

## Soft k-means Clustering

Implement soft k-means clustering by changing the definition of the responsibility vector such that:

1. Each point belongs to a mean as a function of the distance between them (closer the mean, higher responsibility)
2. The sum of each mean's ownership per each data point is equal to 1.

In the lecture notes, the responsibility vector for soft k-means clustering is defined as:

$$r_k^{(n)}=\frac{\exp\left(-\beta\,d\left(\vec{m}^{(k)},x^{(n)}\right)\right)}{\sum_{k'}{\exp\left(-\beta\,d\left(\vec{m}^{(k')},x^{(n)}\right)\right)} }$$ 

where $\beta$ is the *softness parameter* $(\beta\rightarrow\infty\Rightarrow\text{Hard k-means algorithm.})$ and $d\left(\vec{m}^{(k)},x^{(n)}\right)$ is the distance function between the k<sup>th</sup> mean and the n<sup>th</sup> data point.

Define your responsibility vector using a different function than the one given and use it to cluster the data given in {download}`Exercise_04_b.csv<../data/Exercise_04_b.csv>`

```{code-cell} ipython3

```
