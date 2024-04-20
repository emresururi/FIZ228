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

# Lecture Notes

+++

1. [Importing, parsing, processing and exporting datasets](FIZ228_01_DataProcesses)
    * Usage of the Pandas library to import data from CSV files
    * Working with Pandas dataframe
    * **Case Study:** Processing a Meteorological Dataset
2. [Visualization of datasets](FIZ228_02_DataVisualization)
    * Showcase of the Seaborn plotting library
3. [Least Squares Method & Error Estimations](FIZ228_03_LeastSquaresErrors)
    * **Case Study:** Free Fall
    * Types of Error Estimations
    * Fitting through minimization of least squares error
    * Coefficient of Determination ($r^2$)
    * scipy.optimize.minimize, numpy.linalg.lstsq, scipy.linalg.lstsq, scipy.optimize.least_squares, 
4. [Regression](FIZ228_04_Regression)
    * **Case Study:** Drag Force
    * Least-squares Method
    * Adaptation of the Least-squares to non-linear models
    * **Case Study:** FTIR data of Silica
    * np.polyfit, scipy.optimize.curve_fit
5. [Interpolation](FIZ228_05_Interpolation)
    * Polynomial Interpolation
    * Newton Interpolating Polynomials
    * Lagrange Interpolating Polynomials
    * Inverse Interpolation
    * _Bonus: Finite Difference Method_
    * **Example:** Heat distribution of a rod with boundary conditions
    * Polyfit, poly1d, polyval and poly + roots
6. [Minimization & Optimization](FIZ228_06_MinimizeOptimize)
    * Single variable function
    * Multi-variate function
    * Minimization with constraints
    * **Example:** Heron's Formula for Triangle's Area
    * Gradient Descent Algorithm
    * **Case Study:** 2 Springs, 1 Mass, 1 Side
7. [Clustering and Classification](FIZ228_07_ClusteringAndClassification)
    * Advantages of Clustering
    * k-means Clustering
8. [Ordinary Differential Equations](FIZ228_08_OrdinaryDifferentialEquations)
    * Finite Difference Method
    * Euler's Method
    * Runge-Kutta Method (4th order: RK4)
    * ODEs with initial conditions
    * ODEs with boundary conditions
