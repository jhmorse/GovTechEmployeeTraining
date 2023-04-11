# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:14:43 2023

@author: john.h.morse@gmail.com
"""

#%%
# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Set working directory
curr_dir = os.getcwd()
#os.chdir(curr_dir)

#%%

# Path where files are located
path = 'C:/Users/jomors/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/Data/'

## GovTech GTMI Scoring data
# Open Excel data file
#filename = '../Data/WBG_GovTech Dataset_Oct2022.xlsx'
filename = 'WBG_GovTech Dataset_Oct2022.xlsx'
govtech_raw = pd.read_excel(path + filename, sheet_name = 'CG_GTMI_Data', nrows=396)

# Remove 2020 data
govtech_2022 = govtech_raw[govtech_raw.Year == 2022]

# Select columns of interest
govtech = govtech_2022[['Code', 'Economy', 'Population', 'Level', 'Reg', 
                        'Grp', 'GTMI', 'CGSI', 'PSDI', 'DCEI', 'GTEI',
                        'I-45', 'I-45.4', 'I-45.5', 'I-45.5.1', 'I-45.5.3',
                        'I-45.6', 'I-45.7']]
# Rename columns
colnames = {'Economy':'Country', 'Level':'IncomeLevel', 'Reg':'Region',
            'Grp':'Group', 'I-45':'DS_Strategy_Program', 'I-45.4':'FocusArea',
            'I-45.5':'DSProgram', 'I-45.5.1':'DSProgramType', 
            'I-45.5.3':'DSProgramMandatory', 'I-45.6':'DSProgramExternal',
            'I-45.7':'DSProgramPublished'}
govtech = govtech.rename(columns = colnames)

govtech.head()

#%%

## GovTech Projects Data
#filename = '../Data/WBG_DG-GovTech_Projects_Oct2022.xlsx'
filename = 'WBG_DG-GovTech_Projects_Oct2022.xlsx'
projects_raw = pd.read_excel(path + filename, sheet_name = 'DG Projects', nrows=1449)

# Select columns of interest
projects = projects_raw[['Project ID', 'Region', 'Country', 'ICR Out', 'IEG Out']]
# Rename columns
colnames = {'Project ID':'ProjectID', 'ICR Out':'ICROutcome', 'IEG Out':'IEGOutcome'}
projects = projects.rename(columns = colnames)

# Remove any projects that do not have either an ICR rating or an IEG rating
# Create separate project files
projectsICR = projects[['ProjectID', 'Region', 'Country', 'ICROutcome']]
projectsIEG = projects[['ProjectID', 'Region', 'Country', 'IEGOutcome']]

# Drop missing values form each
projectsICR = projectsICR.dropna()
projectsIEG = projectsIEG.dropna()

# Put them back together with a full outer join
projects = projectsICR.merge(projectsIEG, on=('ProjectID', 'Region', 'Country'),
                                  how='outer', suffixes=('_ICR', '_IEG'))

projects.head()

#%%

## GDP Data
#filename = '../Data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4770391.csv'
filename = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4770391.csv'
gdp_raw = pd.read_csv(path + filename, header=2)

# Select columns of interest
gdp = gdp_raw[['Country Name', 'Country Code', '2021']]
# Rename columns
gdp = gdp.rename(columns={'Country Name':'Country', 'Country Code':'Code', '2021':'GDP2021'})

gdp.head()

#%%

##############################################################
## Provide some summary statistics

# GovTech Numeric
govtech.GTMI.describe()
govtech.GTMI.median()
govtech.CGSI.describe()
govtech.CGSI.median()
govtech.PSDI.describe()
govtech.PSDI.median()
govtech.DCEI.describe()
govtech.DCEI.median()
govtech.GTEI.describe()
govtech.GTEI.median()

# GDP Numeric
gdp.GDP2021.mean()
gdp.GDP2021.median()
gdp.GDP2021.std()
gdp.GDP2021.min()
gdp.GDP2021.max()

# Non-numeric
govtech.Code.count()
projects.ProjectID.count()

# Count of various responses for categorical/numerical values
govtech.DS_Strategy_Program.value_counts()
govtech.FocusArea.value_counts()
govtech.DSProgramType.value_counts()

projects.ICROutcome.value_counts()
projects.IEGOutcome.value_counts()

#govtech_raw.plot(x='GTEI', y='DCEI', kind='scatter')
#plt.show()

#govtech_raw.plot(x='GTEI', y='PSDI', kind='scatter')
#plt.show()

#govtech_raw.plot(x='GTEI', y='CGSI', kind='scatter')

#govtech_raw.plot(x='GTEI', y='GTMI', kind='scatter')

#govtech_hci = govtech_raw.loc[govtech_raw['I-44'] != '-']
#govtech_hci.plot(x='GTEI', y='I-44', kind='scatter')

#govtech_raw.plot(x='GTEI', y='I-45', kind='scatter')
