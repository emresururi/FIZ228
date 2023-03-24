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

# Visualization of datasets
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

+++

It's always beneficial to check the data before and after we process it as it can offer some hidden relations or the picking of off values. Even though the `matplotlib` module offers elasticity, unfortunately it is not known for its practicality. Wrappers like the `seaborn` module provide functionality with ease.

+++

## "El Clasico"

Let's try to do it old way, using numpy & matplotlib. As we have observed in our previous lecture, pandas were the go-to module when dealing with datasets, but for reference purposes, we'll start with numpy arrays. As numpy arrays can not (by default) store elements of different types, our string timestamps are lost in import.

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
data_np = np.genfromtxt("data/01_meteoblue_Basel_20230303T060433.csv", delimiter=',',
                        filling_values=0.0,skip_header=10)
data_np
```

```{code-cell} ipython3
data_np.shape
```

We're going to implement meaningful indexes as the first column, by joining the year, month, day index with the hour.

Checking the timestamp of the top entries, we see that it goes from '20220101T0000' to '20230303T2300' (with most of the last entries being blank but we'll deal with it later).

```{code-cell} ipython3
flag_break = False
for y in range(22,24):
    if(flag_break):
        break
    for m in range(1,13):
        if(flag_break):
            break
        for d in range(1,32):
            if(flag_break):
                break
            if((m==2) & (d>28)):
                continue
            if((m in [2,4,6,9,11]) & (d>30)):
                continue
            for h in range (0,24):
                print('{:2d}{:02d}{:02d}{:02d}'.format(y,m,d,h))
                date = '{:2d}{:02d}{:02d}{:02d}'.format(y,m,d,h)
                if(date == '23030323'):
                    flag_break = True
                    break
```

```{code-cell} ipython3
i = 0
flag_break = False
for y in range(22,24):
    if(flag_break):
        break
    for m in range(1,13):
        if(flag_break):
            break
        for d in range(1,32):
            if(flag_break):
                break
            if((m==2) & (d>28)):
                continue
            if((m in [2,4,6,9,11]) & (d>30)):
                continue
            for h in range (0,24):
                #print('{:2d}{:02d}{:02d}{:02d}'.format(y,m,d,h))
                date = '{:2d}{:02d}{:02d}{:02d}'.format(y,m,d,h)
                data_np[i,0] = date
                i += 1
                if(date == '23030323'):
                    flag_break = True
                    break
print(i)
data_np
```

```{code-cell} ipython3
data_np[-1000:,0]
```

Let's get rid of those without any temperature information (col #1):

```{code-cell} ipython3
data_np[data_np[:,1] == 0,:]
```

Turns out that from Feb 24th, 2023 and forward, so:

```{code-cell} ipython3
a = data_np.copy()
a
```

```{code-cell} ipython3
a = np.delete(a,np.arange(0,a.shape[0])[a[:,0]>23022400],0)
```

```{code-cell} ipython3
a.shape[0]
```

```{code-cell} ipython3
np_data = a.copy()
```

... and here comes the basic plot:

```{code-cell} ipython3
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
data_2022 = data_np[data_np[:,0]<23010100,:]
data_2022[-10:,:]
```

```{code-cell} ipython3
plt.plot(data_2022[:,0],data_2022[:,1],"b-s")
plt.title("Graph via Matplotlib")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()
```

```{code-cell} ipython3
plt.plot(data_2022[:,1])
plt.show()
```

```{code-cell} ipython3
filter_1 = (data_np[:,0]>=23010100) & (data_np[:,0]<23020100)
plt.plot(data_np[filter_1,0],data_np[filter_1,1],"b-s")
plt.title("Graph via Matplotlib")
plt.xlabel("January 2023")
plt.ylabel("Temperature")
plt.show()
```

## Exporting a numpy array as a CSV file

While we are at it, here is how we can export a numpy array as CSV:

```{code-cell} ipython3
np.savetxt('del_this_file.csv', data_np, delimiter = ",")
```

##  Importing a CSV file with Pandas

Now that we have experienced the pains of the "old" method, let's revive the technique we have acquired last week: using `Pandas` to hold the data in a dataframe!

```{code-cell} ipython3
import pandas as pd
```

```{code-cell} ipython3
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
data1 = pd.read_csv("data/01_meteoblue_Basel_20230303T060433.csv",
                                         skiprows=9)
data1.columns = ['Timestamp','Temperature','Relative Humidity',
                 'Cloud Coverage', 'Sunshine Duration','Radiation']
data1 = data1.set_index('Timestamp')
data1
```

Even though, it is completely possible to plot dataframe using matplotlib there's actually a much better way to do it: enter the seaborn module!

```{code-cell} ipython3
import seaborn as sns
sns.set_theme() # To make things appear "more cool" 8)
```

```{code-cell} ipython3
data1.loc[:,"Relative Humidity"].max()
```

```{code-cell} ipython3
data1.loc[:,"Sunshine Duration"].max()
```

```{code-cell} ipython3
filter_202208w1 = ((data1.index>="20220801") & 
                 (data1.index<"20220808"))
data_202208w1 = data1.loc[filter_202208w1].copy()
data_202208w1
```

```{code-cell} ipython3
data_202208w1.shape
```

Here, it's as simple as it gets! We are just letting seaborne to figure out what we need:

```{code-cell} ipython3
plt1 = sns.relplot(data=data_202208w1)
```

## Plotting a specific column
We can easily designate columns to be used for the x & y parameters for our graph:

```{code-cell} ipython3
plt2 = sns.relplot(data=data_202208w1,x="Temperature",y="Relative Humidity")
```

And here is a beauty: by `hue` and `size` parameters, we can classify using other column values, making it easier to investigate the dependencies wrt these columns:

```{code-cell} ipython3
plt3 = sns.relplot(data=data_202208w1,x="Temperature",y="Relative Humidity",
                  hue="Temperature",size="Relative Humidity")
```

And this is our attempt to further classify things by adding the `style` alas it kind of fails

```{code-cell} ipython3
plt3 = sns.relplot(data=data_202208w1,x="Temperature",y="Relative Humidity",
                  style="Temperature")
```

Seems that it doesn't like so many classification wrt the values. Luckily we can work around it, by _smoothing_ things out! 8)

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
data_202208w1
```

```{code-cell} ipython3
print("T_min: {:.6f}C | T_max: {:.3f}C"
      .format(data_202208w1.Temperature.min(),data_202208w1.Temperature.max()))
```

```{code-cell} ipython3
data_202208w1[data_202208w1.Temperature == data_202208w1.Temperature.min()]
```

```{code-cell} ipython3
print(data_202208w1.index[data_202208w1.Temperature == data_202208w1.Temperature.min()][0])
```

```{code-cell} ipython3
data_202208w1.Temperature/10
```

```{code-cell} ipython3
np.floor(data_202208w1.Temperature / 10.0) * 10
```

```{code-cell} ipython3
data_202208w1.Temperature
```

Here we add a new column `TempFloor` that stores the smoothed out temperature values:

```{code-cell} ipython3
tempsf = np.floor(data_202208w1.loc[:,"Temperature"] / 10.0) * 10
data_202208w1.loc[:,"TempFloored"] = tempsf.loc[:]
```

```{code-cell} ipython3
data_202208w1
```

```{code-cell} ipython3
plt4 = sns.relplot(data=data_202208w1,x="Temperature",y="Relative Humidity",
                  style="TempFloored")
```

Enough with the scatter plots, lets connect the dots with the `kind` parameter:

```{code-cell} ipython3
plt4 = sns.relplot(data=data_202208w1,x="Timestamp",y="Temperature", 
                   kind="line", marker="^")
```

Here is the same thing without the markers:

```{code-cell} ipython3
plt4 = sns.relplot(data=data_202208w1,x="Timestamp",y="Temperature", 
                   kind="line")
```

```{code-cell} ipython3
data_202208w1
```

Let's further classify such that those entries with their humidity above the mean value will be labeled as "humid", whereas those below will be "dry". 

Therefore, we have to start with calculating the mean:

```{code-cell} ipython3
data_202208w1["Relative Humidity"].mean()
```

and we define a new column for the job:

```{code-cell} ipython3
data_202208w1['RHClass'] = 0
```

```{code-cell} ipython3
data_202208w1
```

How do we single out the ones that have their humidity above the average? By filtering of course! 8)

```{code-cell} ipython3
filter_2 = data_202208w1['Relative Humidity']>52
```

```{code-cell} ipython3
data_202208w1.loc[filter_2,'RHClass']
```

```{code-cell} ipython3
filter_2
```

```{code-cell} ipython3
np.invert(filter_2)
```

So, we fill the 'RHClass' column of the ones above the mean with "humid"; and with "dry" for the others (please observe how we invert the booleans with the "invert").

```{code-cell} ipython3
data_202208w1.loc[filter_2,'RHClass'] = 'humid'
data_202208w1.loc[np.invert(filter_2),'RHClass'] = 'dry'
```

```{code-cell} ipython3
data_202208w1
```

```{code-cell} ipython3
plt5 = sns.relplot(data=data_202208w1,x="Timestamp",y="Temperature", kind="line", 
                   style="RHClass", hue="RHClass")
```

```{code-cell} ipython3
(plt5.map(plt.axhline,y = 22.5, color=".5", dashes=(2, 1), zorder=0)
.set_axis_labels("Day Hour", "Temperature")
.fig.suptitle("Test Graph"))
```

```{code-cell} ipython3
plt5 = sns.relplot(data=data_202208w1,x="Timestamp", y="Temperature", 
                   kind="line", col="RHClass")
```

## Histogram Plots
Histogram bars are also essential - especially if we are dealing with distributions.

```{code-cell} ipython3
plt6 = sns.displot(data=data_202208w1,x="Temperature",
                   col="RHClass",bins=10)
```

```{code-cell} ipython3
data_g = np.random.normal(0,10,1000)
```

```{code-cell} ipython3
data_g
```

```{code-cell} ipython3
plt_gauss = sns.displot(data_g,bins=20,color="r",kde=True,rug=True,)
```

# Summary / Practical Case

+++

## 'Old Style' plot parameters

```{code-cell} ipython3
x_val = np.linspace(-4,5,20)
y_val = x_val**2-2*x_val-7

df_xy = pd.DataFrame({'xx':x_val,'yy':y_val})
```

```{code-cell} ipython3
plt_xy = sns.relplot(data=df_xy,x='xx',y='yy')
```

```{code-cell} ipython3
plt_xy = sns.relplot(data=df_xy,x='xx',y='yy',
                     kind="line",marker="d",
                    markersize=9,markerfacecolor="red",
                    markeredgecolor="green",
                    color="gray",linestyle="--",linewidth=3)
plt.xlabel('x values')
plt.ylabel('y values')
plt.title(r'$x^2-2x-7$')
plt.show()
```

## Pretty much all the useful set

```{code-cell} ipython3
N  = 10
data2 = pd.DataFrame(np.empty((N*N,3),int),columns=['x','y','val'])
k = 0
for i in range(N):
    for j in range(N):
        data2.iloc[k,:] = [i,j,np.random.rand()]
        k += 1
data2
```

```{code-cell} ipython3
data2['xymod'] = np.mod(data2.x+data2.y,5)
data2
```

```{code-cell} ipython3
plt2 = sns.relplot(data=data2,x='x',y='y',hue='val',
                       size='val',style='xymod')
#k=plt.legend(bbox_to_anchor=(1.8,1.01),loc='upper right')
#plt.show()
```

```{code-cell} ipython3
plt3 = sns.relplot(data=data2,x='x',y='y',hue='val',
                       size='val',style='xymod',col=np.mod(data2.xymod,2))
plt.show()
```

```{code-cell} ipython3

```
