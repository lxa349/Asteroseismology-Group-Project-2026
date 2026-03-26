#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:53:09 2026

@author: isaac
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import time

plt.style.use('dark_background')

def read_from_csv(filepath):
    dataframe = pd.read_csv(filepath)
    x = dataframe["Freq"].to_list()
    y = dataframe["Power"].to_list()
    # plt.axvline(nu_max, label = "nu_max")
    plt.loglog(x, y, color = "c", label = "Power spectrum")
    plt.title(f"{filepath[63:-4]} power spectrum")
    plt.xlabel("Frequency /μHz")    
    plt.ylabel("Power /ppm^2 μHz^-1")
    plt.legend(loc = 'best', fontsize = 8)
    plt.show()
    # print(dataframe["Freq"][len(dataframe["Freq"])-1])
    
def read_from_csv_osc(filepath, xmin, xmax):
    dataframe = pd.read_csv(filepath)
    x = dataframe["Freq"].to_list()
    y = dataframe["Power"].to_list()
    plt.plot(x, y, color = "c", label = "Power spectrum")
    plt.xlabel("Frequency /μHz")
    plt.ylabel("Power /ppm^2 μHz^-1")
    # plt.ylim(1e-6,3)
    plt.xlim(xmin,xmax)
    plt.title(f"{filepath[71:-4]} power spectrum")
    plt.legend(loc = 'best', fontsize = 8)
    plt.show()
    

# nu_max_list = []
# with open('/home/isaac/Documents/Asteroseismology Project/Asteroseismology-Group-Project-2026-main/starlist/50_stars_NSR_below_20_test/nu_max') as nu_max_file:
#     for line in nu_max_file:
#         nu_max_list.append(line)
        

# for i in range(0, len(nu_max_list)):
#     if i % 2 == 0:
#         nu_max_list[i] = str(nu_max_list[i][:-3])
#     elif i % 2 == 1:
#         nu_max_list[i] = float(nu_max_list[i][9:-3])
        
# # print(nu_max_list)
# final_list = []
# for j in range(0, len(nu_max_list)):
#     if j % 2 == 1:
#         final_list.append(nu_max_list[j])
        

# count = len(nu_max_list)-1
# for filepath in glob.iglob('/home/isaac/Documents/Asteroseismology Project/Asteroseismology-Group-Project-2026-main/starlist/50_stars_NSR_below_20_test/*.csv'):
#     # print(final_list[count])
#     read_from_csv(filepath, nu_max_list[count])
#     print(f"Star: {nu_max_list[count-1]}")
#     print(f"nu_max: {nu_max_list[count]}")
#     count -= 2
#     time.sleep(1)
    

for filepath in glob.iglob('/home/isaac/Documents/Asteroseismology Project/13CuriosityList/*.csv'):
    read_from_csv(filepath)


