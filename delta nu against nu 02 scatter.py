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

def quad_func(x, a, b, d):
    x = np.asarray(x)
    return a*x**2 + b*x + d

def exp_func(x, coeff, exponent, offset):
    x = np.asarray(x)
    return coeff*np.exp(exponent*x) + offset

table = Table.read('/home/isaac/Documents/Asteroseismology Project/asym_fit_table.tex').to_pandas()


nu_max_list = []
nu_02_list = []
for i in range(0, len(table)):
    nu_max_list.append(float(table["numax"][i][1:-1]))
    nu_02_list.append(float(table["dnu02"][i][1:-28]))
 
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
            swaps += 1
    if swaps == 0:
        noswaps = True
    

plt.scatter(nu_max_list, nu_02_list, s = 5, label = "Stars")
# plt.plot(nu_max_list, nu_02_list, label = "Stars")

popt, pcov = curve_fit(lin_func, nu_max_list, nu_02_list)
plt.plot(nu_max_list, lin_func(nu_max_list, *popt), 'b-',
         label='linear fit: m=%5.5f, c=%5.5f' % tuple(popt))

popt, pcov = curve_fit(quad_func, nu_max_list, nu_02_list)
plt.plot(nu_max_list, quad_func(nu_max_list, *popt), 'r-',
         label='quadratic fit: a=%5.5f, b=%5.5f, d = %5.5f' % tuple(popt))

# popt, pcov = curve_fit(exp_func, nu_max_list, nu_02_list)
# plt.plot(nu_max_list, exp_func(nu_max_list, *popt), 'g-',
#          label='exponential fit: A=%5.5f, B=%5.5f, C=%5.5f' % tuple(popt))




plt.xlabel('nu_max values')
plt.ylabel('nu_02 values')
plt.legend(loc = 'best', fontsize = 6)
plt.show()

print(np.linalg.cond(pcov))
print(np.diag(pcov))
