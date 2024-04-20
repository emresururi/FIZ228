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

# Exercise Set #1
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

# 1
Import the meteorological data ({download}`01_meteoblue_Basel_20230303T060433.csv<../data/01_meteoblue_Basel_20230303T060433.csv>`) as a pandas dataframe (if you can't readily access the course webpage, you can also download it from the [meteoblue site](https://www.meteoblue.com/en/weather/archive/export) with the date range being 1/1/2022 - 3/3/2023 and the parameters set to those you'll need for this example ;).

Then:

* Calculate the average daily temperature on February, 5th, 2022
* Find the most cloudy day in January 2022
* Find the most cloudy hour in January 2022
* Find the most sunny day, calculate the total radiation energy received in 2002

+++

# 2

Draw 1000 samples from a Gaussian (normal) distribution with $\mu =  10$, $\sigma = 1.2$

* Calculate the mean and standard deviation of the samples.  
* Plot its histogram for 10 bins (you can use either matplotlib.pyplot or seaborn).  
    _Hint: You can have the bin positions defined as the average of the left and right boundary of each bin_
    ![Exercises_01_histgraph.png](../imgs/Exercises_01_histgraph.png)
* Define a pandas dataframe such that it has three columns:  
    * `bin_pos` : bin position
    * `bin_count` : count of the samples per each bin
    * `bin_dist` : its distance to the mean
* Using seaborn, plot the dataframe data such that the horizontal axis is the bin_pos, the vertical axis is the bin_count and the points' sizes change with respect to bin_dist.
    ![Exercises_01_histgraph_with_size.png](../imgs/Exercises_01_histgraph_with_size.png)
    * _Challenge: try to modify the graph such that the points closer to the mean are displayed bigger in size, while the dots are connected by a line_
    ![Exercises_01_histgraph_with_invsize.png](../imgs/Exercises_01_histgraph_with_invsize.png)

```{code-cell} ipython3

```
