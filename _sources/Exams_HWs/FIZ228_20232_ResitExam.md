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

# Resit Exam (20232)
**FIZ228 - Numerical Analysis**  
Dr. Emre S. Tasci, Hacettepe University

28/06/2024

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
```

**Pick and solve any 3 of the following questions**

_(No bonus will be given for a 4th one!)_

+++

# 1

a. Construct a pandas dataframe with 5 (imaginary) students’ information containing their:    
   <pre>
        1. Name,  
        2. Surname,  
        3. ID #,  
        4. FIZ227 - Programming Letter Grade  
        5. FIZ228 - (Prospective) Final Exam Grade  
        6. Overall Grade Average  </pre>
b. Add your information as the 6th entry  
c. Calculate the average of the "FIZ228 - (Prospective) Final Exam Grade" column and round it  
d. Have the pandas return the information of the student with the highest "Overall Grade Average"  

+++

# 2 

On a 1D wire, some measurements have been made and the following data have been obtained:

|x|y|
|---|---|
|-1|0|
|2|-24|
|4|0|
|6|0|

In the table above, $x$ denotes the distance from a point marked on the wire while $y$ corresponds to the magnitude of the property measured.

Given that the measured property is a continous quantity, estimate:

a) The value at the x = 5 position.  
b) The position where the measured property's value is equal to -75 (within a precision of the order 10<sup>-4</sup>).

_(For the sake of simplicity, units have been ignored.)_

+++

# 3

Suppose that we have two kinds of particles: A and B (you can think of them as golf balls and tennis balls). When they are put in a system, the system's energy is given by the following formula:

$$U(n_A,n_B) = -2n_A^2-n_B^2+5n_An_B+10n_A+70n_B$$

where $n_{\{A,B\}}$ indicates the number of A and B particles in the system.

Due to the system's capacity, and the fact that B particles being bigger than A particles, we have the following restriction:

$$n_A + 2.7n_B \le 101$$

Find the optimal number of A and B particles to be put into the system such that they satisfy the above restriction while yielding the maximum energy.

+++

# 4

Solve the following ODE for the given conditions:

$$y'' + y - e^{-x/10} y' = 0\\y(0)=1,\,y'(0)=0\quad x\in [0,12];\quad h\le0.01$$

Plot your result.
