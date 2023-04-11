# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:14:43 2023

@author: john.h.morse@gmail.com
"""

# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Set working directory
curr_dir = os.getcwd()
#os.chdir(curr_dir)

# Open Excel data file
filename = '../Data/WBG_GovTech Dataset_Oct2022.xlsx'
#govtech_raw = pd.read_excel(filename, sheet_name = 'CG_GTMI_Groups', nrows=198)
govtech_raw = pd.read_excel(filename, sheet_name = 'CG_GTMI_Data', nrows=396)

govtech_raw.head()
govtech_raw.columns

# Remove 2020 data
govtech_2022 = govtech_raw[govtech_raw.Year == 2022]

# Select columns of interest
govtech = govtech_2022[['Code', 'Economy', 'Population', 'Level', 'Reg', 
                        'Grp', 'GTMI', 'CGSI', 'PSDI', 'DCEI', 'GTEI',
                        'I-45', 'I-45.4', 'I-45.5', 'I-45.5.1', 'I-45.5.3',
                        'I-45.6', 'I-45.7']]


govtech_raw.plot(x='GTEI', y='DCEI', kind='scatter')
plt.show()

govtech_raw.plot(x='GTEI', y='PSDI', kind='scatter')
plt.show()

govtech_raw.plot(x='GTEI', y='CGSI', kind='scatter')

govtech_raw.plot(x='GTEI', y='GTMI', kind='scatter')

govtech_hci = govtech_raw.loc[govtech_raw['I-44'] != '-']
govtech_hci.plot(x='GTEI', y='I-44', kind='scatter')

govtech_raw.plot(x='GTEI', y='I-45', kind='scatter')
