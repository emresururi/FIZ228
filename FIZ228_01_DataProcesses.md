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

# Importing, parsing, processing and exporting datasets
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

Numerical analysis deals with all kinds of data: be it experimental results, surveys, collected values, etc. Almost all the time these data are too numerous to be entered or transferred via manual means so, different standards for file formats have been developed for different purposes. Even though every format can include specific notation and data format, most of the programs and applications are compatible with the most basic format of all: the comma-separated-values (CSV) format.

In this format, different measurements / sets of data are stored in rows and for each set, different types are seperated via the comma (or tab or space). Although not standard, definition or extra information can be present at the start and it is usually the case that the row labels are also included most of the time for clarification. In this sense, one can visualize a spreadsheet (like those in MS Excel, LibreOffice Calc or Google Sheets) as a one-to-one representation of a CSV file.

An example is given below:

```csv
2theta/Degree,Intensity_GFO(Ga/Fe=1),Normalised_XRD_different ratio,Ga/Fe=1,Normalised_XRD_different ratio,Ga/Fe=0.5
20,86.06418,20,6.26E-04,20,0.01343
20.01968,96.05973,20.01968,6.99E-04,20.01968,0.00924
20.03936,108.07215,20.03936,7.87E-04,20.03936,0.01144
20.05904,122.78125,20.05904,8.94E-04,20.05904,0.00909
20.07872,141.1535,20.07872,0.00103,20.07872,0.00612
20.0984,164.50312,20.0984,0.0012,20.0984,0.01085
20.11808,194.49239,20.11808,0.00142,20.11808,0.00873
20.13776,233.03232,20.13776,0.0017,20.13776,0.01075
20.15744,282.11963,20.15744,0.00205,20.15744,0.01121
```

{download}`GaFeO3 XRD data (Sun et al.)<data/01_GaFeO3_XRD.csv>`

+++

## Importing a CSV file with Pandas

+++

Pandas is one of the most commonly used modules as it efficiently handles tabulated datas such as those coming from spreadhsheets. It has a direct method for importing from CSV files: `read_csv()`

```{code-cell} ipython3
import numpy as np
import pandas as pd
```

```{code-cell} ipython3
data_XRD = pd.read_csv("data/01_GaFeO3_XRD.csv")
print(data_XRD)
```

Here we see that, the first row has automatically been treated as the header row by Pandas.

```{code-cell} ipython3
data_XRD
```

The output (whether printed out via `print` or directly called) can be seen as summarized to 10 rows (5 from above, 5 from below) for brevity. This behaviour can be changed via 'display' options:

```{code-cell} ipython3
pd.set_option('display.max_rows',6)
print(data_XRD)
data_XRD
```

to display all rows, we set 'display.max_rows' to `None`:

```python
pd.set_option('display.max_rows',None)
```

+++

We can also override it temporarily using the "with" procedure:

```{code-cell} ipython3
with pd.option_context('display.max_rows',None):
    print(data_XRD)
```

### Getting rid of the unnecessary header stuff
Sometimes, the source file can have informational lines at the beginning -- to skip these lines, there is the `skiprows` parameter in `read_csv()`:

```python
# The following will ignore the first 9 lines of the CSV file:
data_wo_clutter = pd.read_csv("filename.csv", skiprows=9)
```

+++

## Column Labels

The column labels can be overwritten:

```{code-cell} ipython3
data_XRD.columns
```

```{code-cell} ipython3
data_XRD.columns = ['2theta','Intensity','Normalized Ratio','Ga/Fe=1',
                    'Normalized Ratio(2)','Ga/Fe=0.5']
```

```{code-cell} ipython3
data_XRD
```

## Index
Index is a unique column that directly addresses the row and its denoted by the bold formatting in the output. By default, Pandas assign a sequential index, but we can also designate an existing column as index via `set_index()` method. As this method creates a new instance, it should be redirected to the original dataframe for in-place replacement:

```{code-cell} ipython3
data_aux = data_XRD.set_index('2theta')
data_aux
```

```{code-cell} ipython3
data_aux.index
```

## Accessing the data

+++

Now that we have our values in the dataframe, we can start using them. Pandas support _classic_ column & row index reference (like NumPy or GNU Octave / MATLAB array access) but the data can also be accessed directly via column and row labels as well.

```{code-cell} ipython3
pd.set_option('display.max_rows',10)
data_XRD
```

### Old-school style: referring via col/row indexes

For this kind of reference, we use the `iloc` command:

```{code-cell} ipython3
data_XRD.iloc[2,0] # 2nd Row, 0th Col
```

```{code-cell} ipython3
data_XRD.iloc[[4,1],[1,0]] # Rows: 4 and 1 && Cols: 1 and 0
```

```{code-cell} ipython3
data_XRD.iloc[1:4,[0,2]] # Rows: [1,4) && Cols: 0 and 2
```

```{code-cell} ipython3
data_XRD.iloc[[1,3,6],:] # Rows: 1, 3 and 6 && All cols
```

### Referring via the col/row labels

We also have the option to call by the labels, using `loc`.

At the moment, our XRD data doesn't have a column for a meaningful index but until we have a more suitable one, let's work on the version with the 2theta defined as the index column:

```{code-cell} ipython3
data_aux
```

```{code-cell} ipython3
data_aux.loc[[20.03936],['Intensity']]
```

```{code-cell} ipython3
data_aux.loc[20.03936,['Intensity']]
```

```{code-cell} ipython3
data_aux.loc[20.03936,'Intensity']
```

```{code-cell} ipython3
data_aux.loc[20.03936,['Intensity','Ga/Fe=0.5']]
```

```{code-cell} ipython3
data_aux.loc[20.03936]
```

```{code-cell} ipython3
data_aux.loc[[20.03936]]
```

`loc` takes the first parameter as the requested rows and the second parameter as the requested columns (both are usually given as lists).

```{code-cell} ipython3
data_aux.loc[[20.03936, 69.96676],
             ['Intensity','Ga/Fe=1','Normalized Ratio']]
```

### Further accessing (by examples)

* **Iterating over via `items`**

```{code-cell} ipython3
for col,dat in data_aux.loc[[20.03936, 69.96676],
             ['Intensity','Ga/Fe=1','Normalized Ratio']].items():
    print(col,"**",*dat[:])
```

* **Converting the result to array via `values`**

```{code-cell} ipython3
data_aux.loc[[20.03936, 69.96676],
             ['Intensity','Ga/Fe=1','Normalized Ratio']].values
```

* **Directly accessing via column name**

```{code-cell} ipython3
data_aux['Intensity']
```

```{code-cell} ipython3
data_aux[['Intensity','Normalized Ratio']]
```

## Filtering

For filtering, we just make a proposition and get the results for each row as True/False Booleans.

First, write the query that will bring us the set that we want to filter. As an example, let's assume that we are interested in the 2theta angles in the range (20,30]: therefore we first write the query that returns the 2theta column:

```{code-cell} ipython3
data_XRD['2theta']
```

then, we make our proposition:

```{code-cell} ipython3
(data_XRD['2theta'] > 20) & (data_XRD['2theta'] <= 30)
```

Here is our filter! So let's define it and use it:

```{code-cell} ipython3
filter1 = (data_XRD['2theta'] > 20) & (data_XRD['2theta'] <= 30)
data_XRD.loc[filter1,:]
```

```{code-cell} ipython3
data_XRD.loc[filter1,['2theta','Intensity','Ga/Fe=1']]
```

We could have also achieved the same result quick and dirty by directly proposing the statement and placing it into the heart of our dataframe:

```{code-cell} ipython3
data_XRD[(data_XRD['2theta'] > 20) & (data_XRD['2theta'] <= 30)]
```

The filtering criteria doesn't necessarily be based on the same column. For example, let's get the entries that has 2theta less than 30<sup>o</sup> with intensities higher than 180:

```{code-cell} ipython3
filter2 = (data_XRD['2theta'] < 30) & (data_XRD['Intensity'] > 180)
data_XRD.loc[filter2,:]
```

What if we wanted to apply a criteria to the row index? Can we specify a criteria for them as well? Let's switch to our data_aux dataframe where we had defined the 2theta column as the index:

```{code-cell} ipython3
data_aux
```

```{code-cell} ipython3
data_aux.index
```

Let's pick the angles that are less than 30<sup>o</sup> _or_ greater than 60<sup>o</sup> degrees:

```{code-cell} ipython3
filter3 = (data_aux.index < 30) | (data_aux.index > 60)
filter3
```

```{code-cell} ipython3
data_aux[filter3]
```

### Statistical stuff

* **max & max location**

```{code-cell} ipython3
data_aux['Intensity'].max()
```

where is it?

```{code-cell} ipython3
data_aux['Intensity'].idxmax()
```

Let's verify:

```{code-cell} ipython3
data_aux.loc[33.22476]
```

* **mean, median, variance, stddev**

```{code-cell} ipython3
data_aux['Intensity'].mean()
```

```{code-cell} ipython3
data_aux['Intensity'].median()
```

```{code-cell} ipython3
data_aux['Intensity'].var()
```

```{code-cell} ipython3
data_aux['Intensity'].std()
```

## Processing

Now that we know how to slice via filtering, we can do whatever we want with the sections of the data we're interested in. For example, let's calculate the standard deviation of the peak around 30<sup>o</sup> (it would be a good idea if we'd plot it first ;)

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
plt.plot(data_aux.index,data_aux['Intensity'])
plt.show()
```

Zoom a little bit in:

```{code-cell} ipython3
subdata_range = (data_aux.index > 28) & (data_aux.index < 35)
```

```{code-cell} ipython3
data_aux.loc[subdata_range,'Intensity']
```

```{code-cell} ipython3
plt.plot(data_aux.index[subdata_range],
         data_aux.loc[subdata_range,'Intensity'])
plt.show()
```

```{code-cell} ipython3
subdata_range = (data_aux.index > 32.5) & (data_aux.index < 34)
plt.plot(data_aux.index[subdata_range],
         data_aux.loc[subdata_range,'Intensity'])
plt.show()
```

```{code-cell} ipython3
np.mean(data_aux.index[subdata_range]),\
np.std(data_aux.index[subdata_range])
```

... why, after all, did we use numpy to calculate the mean and standard deviation of the graph? Because, the x-axis is the index. Otherwise, the statistical methods all work on the data columns:

```{code-cell} ipython3
data_aux.mean()
```

```{code-cell} ipython3
data_aux.std()
```

Moral of the story: Be careful when defining your index.

+++

## Creating a Dataframe

We can directly create a dataframe with the `DataFrame()` command:

```{code-cell} ipython3
# Start with an empty dataframe:
subdata = pd.DataFrame({'TwoTheta' : [],'Intensity' : []})
subdata
```

We can fill it individually via the `concat()` method:

(_concat returns a new dataframe, so if we want to update our present dataframe, we need to act it in-place._)

```{code-cell} ipython3
subdata = pd.concat([subdata,
            pd.DataFrame({'TwoTheta':[30.0],'Intensity':[29000]})])
subdata
```

Repeat the above again, and we'll see that the index is not being updated (i.e., it's being taken as it is):

```{code-cell} ipython3
subdata = pd.concat([subdata,
            pd.DataFrame({'TwoTheta':[30.0],'Intensity':[29000]})])
subdata
```

If we want the index re-evaluated, we must set the `ignore_index` parameter to 'True':

```{code-cell} ipython3
subdata = pd.concat([subdata,
                     pd.DataFrame({'TwoTheta':[30.0],'Intensity':[29000]})]
                    ,ignore_index=True)
subdata
```

We can also force a designated index if we want:

```{code-cell} ipython3
new_data = pd.DataFrame({'TwoTheta':[45.0],'Intensity':35000},index = {"n5"})
new_data
```

```{code-cell} ipython3
subdata = pd.concat([subdata,new_data])
subdata
```

```{code-cell} ipython3
subdata.loc['n5',:]
```

## Adding and Dropping Rows/Cols

We have seen how to add a row via the `concat()` method. What if we want to delete (drop) one or more rows?

Once again consider the `subdata` dataframe:

```{code-cell} ipython3
subdata
```

Before we begin, it's a good idea to back up our dataframe:

```{code-cell} ipython3
subdata_org = subdata
```

To delete, we feed the unwanted rows' index information to the `drop()` method:

```{code-cell} ipython3
subdata.drop([1,'n5'])
```

Notice how _1_ is passed as numerical value and _'n5'_ as string above.

Now here is a surprise:

```{code-cell} ipython3
subdata
```

As can be seen, `drop()` returns a new dataframe. So, if we want to update our dataframe, we need to either feed it back to itself or set the `inplace` keyword to `True`:

```{code-cell} ipython3
subdata = subdata.drop([1,'n5'])
subdata
```

```{code-cell} ipython3
subdata.drop(0,inplace=True)
subdata
```

Before we continue further, let's reset our dataframe:

```{code-cell} ipython3
subdata = subdata_org
subdata
```

We have seen how to add a row using the `concat()` method, what if we want to add a new row? There are two ways to do this:

+++

### Adding a new column via concat()

In this approach, we define a new dataframe, but concatenating it vertically by setting the `axis` keyword to 1:

```{code-cell} ipython3
NewCol = pd.DataFrame({'NewVal':[3.2,4.5,-9.3]})
NewCol
```

```{code-cell} ipython3
pd.concat([subdata,NewCol],ignore_index=True,axis=1)
```

As you can observe, the 'missing' data entries are filled with 'NaN'.

+++

### Adding a new column directly:

We can simply define a 'non-existing' column as if there was and go on with it:

```{code-cell} ipython3
subdata = subdata_org
subdata
```

```{code-cell} ipython3
subdata['NewCol'] = [3.5,6.1,-3.7,4.5]
subdata
```

But in this approach, we must match all the entries.

+++

### Dropping a column

Dropping a column can be done via `drop()` while setting the `axis` to 1:

```{code-cell} ipython3
subdata
```

```{code-cell} ipython3
subdata.drop('NewCol',axis=1)
```

and just like the row dropping, if you want this operation to be done in-place, either direct the output to the source or set the `inplace` option to `True`.

+++

## Inter-operations between columns

Operating with the column values are pretty straightforward: If you can refer to the values of the columns, then you can also operate on them! ;)

```{code-cell} ipython3
subdata
```

Let's add the TwoTheta and NewCol values:

```{code-cell} ipython3
subdata['TwoTheta'] + subdata['NewCol']
```

while we are at it, let's collect the results in a new column:

```{code-cell} ipython3
subdata['Results'] = subdata['TwoTheta'] + subdata['NewCol']
```

```{code-cell} ipython3
subdata
```

## Exporting the DataFrame to a CSV file

```{code-cell} ipython3
data_aux
```

```{code-cell} ipython3
:tags: [output_scroll]

# With the row labels (index) and column labels included:
data_aux.to_csv('out/01_out.csv')
print(data_aux.to_csv())
```

```{code-cell} ipython3
:tags: [output_scroll]

# Without the row labels but with the column labels:
print(data_aux.to_csv(index=False))
```

```{code-cell} ipython3
:tags: [output_scroll]

# Without the row labels and the column labels:
print(data_aux.to_csv(index=False, header=False))
```

```{code-cell} ipython3
:tags: [output_scroll]

# Specify header for the index column:
print(data_aux.to_csv(index_label='Two Theta'))
```

```{code-cell} ipython3
:tags: [output_scroll]

# Use ';' as the seperator, instead of ','
data_aux.to_csv(sep=';',path_or_buf='out/01_out.csv')
print(data_aux.to_csv(sep=';'))
```

# Sample Case: Meterological Dataset

[meteoblue.com](https://www.meteoblue.com/en/weather/archive/export?daterange=2022-01-01%20-%202023-03-03&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&min=2022-01-01&max=2023-03-03&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=relhum2m&params%5B%5D=&params%5B%5D=&params%5B%5D=totalClouds&params%5B%5D=&params%5B%5D=sunshine&params%5B%5D=swrad&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=1&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30) site offers historical data for the city of Basel. The dataset we are going to use contains the data for 1/1/2022 - 3/3/2023.

+++

Checking the downloaded CSV file, we see that it has a header section before the data section begins:

+++

```
  1: location,Basel,Basel,Basel,Basel,Basel
  2: lat,47.75000,47.75000,47.75000,47.75000,47.75000
  3: lon,7.50000,7.50000,7.50000,7.50000,7.50000
  4: asl,363.653,363.653,363.653,363.653,363.653
  5: variable,Temperature,Relative Humidity,Cloud Cover Total,Sunshine Duration,Shortwave Radiation
  6: unit,°C,%,%,min,W/m²
  7: level,2 m elevation corrected,2 m,sfc,sfc,sfc
  8. resolution,hourly,hourly,hourly,hourly,hourly
  9: aggregation,None,None,None,None,None
 10: timestamp,Basel Temperature [2 m elevation corrected],Basel Relative Humidity [2 m],Basel Cloud          Cover Total,Basel Sunshine Duration,Basel Shortwave Radiation
 11: 20220101T0000,6.0602455,94.19782,1.5,0.0,0.0
 12: 20220101T0100,5.5602455,94.83262,3.0,0.0,0.0
 13: 20220101T0200,4.630245,96.47203,0.3,0.0,0.0
 14: 20220101T0300,3.6602454,97.1187,1.2,0.0,0.0
 15: 20220101T0400,3.8802452,96.16575,1.0,0.0,0.0
```

_(the line numbers are added for clarity - they are not present in the CSV file)_

The data actually begins at the 11th line while the 10th line contains the labels. So, we need to skip the first 9 lines while importing the CSV file to Pandas.

```{code-cell} ipython3
data_weather = pd.read_csv("data/01_meteoblue_Basel_20230303T060433.csv",skiprows=9)
data_weather
```

As the timestamp column is unique, it will be a good idea to designate it as the index:

```{code-cell} ipython3
data_weather = data_weather.set_index('timestamp')
data_weather
```

Also, the column labels look a bit crowded, so let's relabel them as well:

```{code-cell} ipython3
data_weather.columns = ['Temperature','Humidity','CloudCoverage',
                        'SunshineDuration','Radiation']
data_weather
```

As today's (3/3/2023) data hasn't been entered yet, we have blanks there. Let's also get rid of today's data. We do the row deletion via `drop()` method where it takes the index of the rows that will be deleted ('dropped'):

```{code-cell} ipython3
# This filters today's entries
data_weather[data_weather.index>'20230303']
```

```{code-cell} ipython3
# Here is the index of the filtered entries
data_weather[data_weather.index>'20230303'].index
```

```{code-cell} ipython3
# Now that we have the index, we can feed them to drop to delete them
data_weather = data_weather.drop(data_weather[data_weather.index>'20230303'].index)
data_weather
```

Pay attention that `drop()` method returns a new dataframe. If we want to update our current dataframe, we need to use it in-place.

But wait! We still have the 'NaN' values in the preceeding days -- let's completely find out which days those are:

```{code-cell} ipython3
# as NaN value indicates null values,
# we filter via the 'isnull()' method
data_weather[data_weather.Temperature.isnull()]
```

```{code-cell} ipython3
# So, let's get rid of them totally:
data_weather = data_weather.drop(data_weather[data_weather.Temperature.isnull()].index)
data_weather
```

## Exercises
* Calculate the average daily temperature on February, 5th
* Find the most cloudy day in January
* Find the most cloudy hour in January
* Find the most sunny day, calculate the total radiation energy received

+++

## References
* X. Sun, D. Tiwari, D.J. Fermin "High Interfacial Hole-Transfer Efficiency at GaFeO<sub>3</sub> Thin Film Photoanodes" Adv. Energy Materials (10) 45 2002784 (2020)  
[DOI: 10.1002/aenm.202002784](https://doi.org/10.1002/aenm.202002784) | [Data](https://data.bris.ac.uk/data/dataset/c4w8vwn8xfr2kozw9k8lb7o1)
* [meteoblue.com - Historical Weather Data for Basel](https://www.meteoblue.com/en/weather/archive/export?daterange=2022-01-01%20-%202023-03-03&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&min=2022-01-01&max=2023-03-03&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=relhum2m&params%5B%5D=&params%5B%5D=&params%5B%5D=totalClouds&params%5B%5D=&params%5B%5D=sunshine&params%5B%5D=swrad&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=1&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30)
