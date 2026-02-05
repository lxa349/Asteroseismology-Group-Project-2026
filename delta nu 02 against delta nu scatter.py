#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:50:03 2026

@author: isaac
"""

'''
Linear fit: dnu_02 = 0.002 * dnu + 2.128
dnu_13 = (5/3) * dnu_02 (Chaplin and Miglio)
'''


from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lin_func(x, m, c):
    return m * np.asarray(x) + c

'''
def quad_func(x, a, b, d):
    x = np.asarray(x)
    return a*x**2 + b*x + d

def exp_func(x, coeff, exponent, offset):
    x = np.asarray(x)
    return coeff*np.exp(exponent*x) + offset

'''

def list_generation():
    # reads from the Kepler LEGACY lists - !CHANGE FILE PATH!
    # it uses versions of the tables which I manually edited, which I should have uploaded to the github
    # Gets nu_max, dnu_02 and Teff data from the tables
    
    table = Table.read('/home/isaac/Documents/Asteroseismology Project/asym_fit_table.tex').to_pandas()
    
    teff_table = Table.read('/home/isaac/Documents/Asteroseismology Project/target_table.tex').to_pandas()
    
    
    nu_max_list = []
    nu_02_list = []
    temp_list = []
    for i in range(0, len(table)):
        nu_max_list.append(float(table["dnu"][i][1:-1]))
        nu_02_list.append(float(table["dnu02"][i][1:-28]))
        temp_list.append(float(teff_table["Teff"][i][1:-1]))
     
    noswaps = False
    while noswaps == False:
        swaps = 0
        for count in range(0, len(nu_max_list)-1):
            if nu_max_list[count] > nu_max_list[count+1]:
                temp = nu_max_list[count]
                nu_max_list[count] = nu_max_list[count+1]
                nu_max_list[count+1] = temp
                temp = nu_02_list[count]
                nu_02_list[count] = nu_02_list[count+1]
                nu_02_list[count+1] = temp
                temp = temp_list[count]
                temp_list[count] = temp_list[count+1]
                temp_list[count+1] = temp
                swaps += 1
        if swaps == 0:
            noswaps = True
            
    return nu_max_list, nu_02_list, temp_list

def linear_curve_fit(func, x, y):
    # uses curve_fit to fit a linear function to the data
    popt, pcov = curve_fit(func, x, y)
    return popt, pcov
    
        
def star_plotting(nu_max_list, nu_02_list, temp_list, popt, pcov, std = 0):    
    # plots a scatter graph of the stars from the LEGACY data, the linear fit, and shades 1std and 2std out from the fit.
    # stars are colour-coded based on effective temperature
    
    if std != 0:

        plt.fill_between(nu_max_list, lin_func(nu_max_list, *popt) - std, lin_func(nu_max_list, *popt) + std, alpha = 0.2, color = "cyan", label = "1std")
        plt.fill_between(nu_max_list, lin_func(nu_max_list, *popt) + 2*std, lin_func(nu_max_list, *popt) - 2*std, alpha = 0.05, color = "green", label = "2std")
    
    plt.scatter(nu_max_list, nu_02_list, s = 5, label = "Stars", c = temp_list, cmap = 'hot')
    plt.colorbar(label = "Effective temperature /K")
    
    plt.plot(nu_max_list, lin_func(nu_max_list, *popt), 'b-',
             label='linear fit: m=%5.5f, c=%5.5f' % tuple(popt))
    
    plt.xlabel('Large Frequency Separation /μHz')
    plt.ylabel('Small Frequency Separation (l = 0,2) /μHz')
    plt.title("Kepler LEGACY stars")
    
    if std != 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc = 'best', fontsize = 6)
    else:
        plt.legend(loc = 'best', fontsize = 6)
    plt.show()
    
    # print(np.linalg.cond(pcov))
    # print(np.diag(pcov))
    print(f"Minimum temperature: {min(temp_list)}\nMaximum temperature: {max(temp_list)}\nTemperature range: {max(temp_list) - min(temp_list)}")
    
    return popt

def calc_residuals(nu_max_list, nu_02_list, grad, intercept):
    # This code calculates standard deviation of residuals
    residuals = []
    for i in range(0, len(nu_max_list)):
        residuals.append(nu_02_list[i] - ((grad * nu_max_list[i]) + intercept))
    
    
    residuals_std = 0
    for k in range(0, len(residuals)):
        residuals_std += (residuals[k]**2)
    residuals_std = np.sqrt(residuals_std/(len(residuals)-2))
    print(f"The standard deviation of the residuals is {residuals_std}.")
    
    return residuals_std

def gaussian_select(value, std):
    # this function produces a normal distribution to draw values from based on a provided mean and standard deviation
    return np.random.normal(value, std)

def find_small_freq_sep(nu_max, grad, intercept, std):
    # this function uses gaussian_select to produce a randomised value for dnu_02 based on an input nu_max value, as well as fit data and residual standard deviation
    initial_02 = (grad * nu_max) + intercept
    final_02 = gaussian_select(initial_02, std)
    return final_02

def main():
    nu_max, nu_02, temp = list_generation()
    popt, pcov = curve_fit(lin_func, nu_max, nu_02)
    print(f"The gradient of the fitted function is {popt[0]}, and the intercept is {popt[1]}.")
    residuals_std = calc_residuals(nu_max, nu_02, popt[0], popt[1])
    star_plotting(nu_max, nu_02, temp, popt, pcov, residuals_std)
    
    # I currently have it finding values for a numax of 1750 microhertz
    print(find_small_freq_sep(1750, popt[0], popt[1], residuals_std))
    
    

main()
