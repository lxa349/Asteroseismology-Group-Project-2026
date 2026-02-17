# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:49:55 2026

@author: lxa349
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.table import Table
import astropy.units as u
import csv

star_name = 'Gaia DR3 5529801905785076992'
model = 'karoff'

def single_import(pic, starname): #if you want a single star: for testing mostly
    pic_list = glob.glob(pic)
    tbl = Table.read(pic_list[0])
    pic_tabl2 = tbl[['StarName', 'Mass', 'Radius', 'Teff']]
    
    for i in pic_tabl2:
        if i[0] == starname:
            return(i[0], i[1], i[2], i[3]) #appends: Name, Mass, Radius, Teff of star named
        
def list_import(file_input):
    attribute_list = []
    
    
    with open(f"{file_input}", 'r') as myfile:
        r = csv.reader(myfile)
        for row in r:
            attribute_list.append([row[0], row[1], row[2], row[3]])
        
    return attribute_list #NOTE: first entry is the column headers, so when reading from this you should skip 1

pic = r"LOPS2PICtarget2.1.0.1-t-fg-c-scv.fits" #path to PIC fits file
ppm = u.def_unit('ppm')

print(list_import('Adjusted final list.csv')) #name of the input csv file


#Output

with open(f"{star_name}.csv", 'w') as myfile: #note the outputted file is large!
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(star.freq_powerspectrum_uHz)
    wr.writerow(granulation_component_theory + facule_component_theory)