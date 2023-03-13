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
curr_dir = 'C:/Users/jomors/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/'
os.chdir(curr_dir)

# Open Excel data file
filename = 'Data/WBG_GovTech Dataset_Oct2022.xlsx'
govtech_raw = pd.read_excel(curr_dir + filename, sheet_name = 'CG_GTMI_Groups', nrows=198)

govtech_raw.head()
govtech_raw.columns

govtech_raw.plot(x='GTEI', y='DCEI', kind='scatter')
plt.show()

govtech_raw.plot(x='GTEI', y='PSDI', kind='scatter')
plt.show()

govtech_raw.plot(x='GTEI', y='CGSI', kind='scatter')

govtech_raw.plot(x='GTEI', y='GTMI', kind='scatter')

govtech_hci = govtech_raw.loc[govtech_raw['I-44'] != '-']
govtech_hci.plot(x='GTEI', y='I-44', kind='scatter')

govtech_raw.plot(x='GTEI', y='I-45', kind='scatter')
