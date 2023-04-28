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
import seaborn as sns

# Set working directory
curr_dir = os.getcwd()
#os.chdir(curr_dir)

#%%

# Path where files are located
path = 'C:/Users/jomors/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/Data/'
path = 'C:/Users/johnh/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/Data/'
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
# Several of the columns get read in as objects instead of numeric.
# Need to convert these to numeric for evaluation.
govtech['DSProgram'] = pd.to_numeric(govtech.DSProgram)
govtech['DSProgramType'] = pd.to_numeric(govtech.DSProgramType)
govtech['DSProgramMandatory'] = pd.to_numeric(govtech.DSProgramMandatory)
govtech['DSProgramExternal'] = pd.to_numeric(govtech.DSProgramExternal)
govtech['DSProgramPublished'] = pd.to_numeric(govtech.DSProgramPublished)
govtech.info()

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

#%%

###########################################################
### Prep GTMI Data

# Merge the GDP data into the GovTech GTMI data.
govtech_gdp = govtech.merge(gdp, on=('Code'), how='left', suffixes=('', '_gdp'))

# Drop the redundant Country value brought into the data set
govtech_gdp = govtech_gdp.drop(columns=['Country_gdp'])

# Did the merge result in any GTMI countries missing a GDP value?
print(np.mean(govtech_gdp.GDP2021 <= 0))

# Show results
govtech_gdp.head()

#%%

# Define the categories and corresponding codes
categories = ['L', 'LM', 'UM', 'H']
codes = [0, 1, 2, 3]

# Create a categorical data type to convert the existing variable
from pandas.api.types import CategoricalDtype
cat_dtype = CategoricalDtype(
    categories=['L', 'LM', 'UM', 'H'], ordered=True)

# Convert the IncomeLevel column to categorical
govtech_gdp['IncomeLevel'] = govtech_gdp['IncomeLevel'].astype(cat_dtype)

# Outcome should look no different
print(govtech_gdp['IncomeLevel'].head(10))

# But the type of the column is now categorical
print(type(govtech_gdp.IncomeLevel.array))

#%%

############################################################
### Prep Projects data

# Distribution of ICROutcome values
category_order = ['HU', 'U', 'MU', 'MS', 'S', 'HS']
sns.catplot(x='ICROutcome', data=projects, kind='count', order=category_order, color='teal')
plt.show()


# Distribution of IEGOutcome values
sns.catplot(x='IEGOutcome', data=projects, kind='count', order=category_order, color='navy')
plt.show()

#%%

# Create binary values for ICROutcome and IEGOutcome
projects['ICROutcomeB'] = np.where(projects['ICROutcome'].isin(['HS', 'S', 'MS']), 1, 0)
projects['IEGOutcomeB'] = np.where(projects['IEGOutcome'].isin(['HS', 'S', 'MS']), 1, 0)

# Check the totals for each binary variable
print(np.sum(projects.ICROutcomeB))
print(np.sum(projects.IEGOutcomeB))

# Display the first few columns of the data set
projects.head(10)

#%%

# Create a set of Country names and Country Codes from the govtech data.
country_lookup = govtech[['Country', 'Code']]

# Join this to the projects data based on the Country name
projectscd = projects.merge(country_lookup, on='Country', how='left', suffixes=('_prj', '_gt'))

# preview the data
projectscd.head(10)

#%%

# Which countries did not get picked up?
projectscd[pd.isna(projectscd['Code'])]['Country'].unique()

# Manually create a lookup data frame with the additional values
add_lookup = pd.DataFrame({'Country':['Turkiye', 'Lao People\'s Democratic Republic', 'Cote d\'Ivoire',
                                   'Egypt, Arab Republic of', 'Central Africa', 'Africa', 'Congo, Democratic Republic of',
                                   'Congo, Republic of', 'Eastern and Southern Africa', 'Yemen, Republic of',
                                   'Sao Tome and Principe', 'Gambia, The', 'Western and Central Africa',
                                   'Venezuela, Republica Bolivariana de', 'Caribbean', 'Iran, Islamic Republic of',
                                   'OECS Countries'], 
                        'Code':['TUR', 'LAO', 'CIV', 'EGY', 'CAF', 'UNK', 'COD', 'COG', 'UNK', 'YEM', 
                                'STP', 'GMB', 'UNK', 'VEN', 'UNK', 'IRN', 'UNK']})

# Let's add these new values to the original lookup data frame
country_lookup_full = pd.concat([country_lookup, add_lookup])

# Repeat the previous merge effort with the longer list of country codes
projectscd = projects.merge(country_lookup_full, on='Country', how='left', suffixes=('_prj', '_add'))

# Validate we no longer have any missing Codes
print(np.sum(pd.isna(projectscd.Code)))

# preview the data
projectscd.head(10)

#%%

print('Total projects: ' + str(len(projectscd)))
print('Projects with unknown countries: ' + str(np.sum(projectscd.Code == 'UNK')))

# Drop the projects with Unknown country codes
projectscd = projectscd[projectscd['Code'] != 'UNK']
print('Remaining projects: ' + str(len(projectscd)))

#%%
# Merge the projects and govtech data
projects_gtmi = projectscd.merge(govtech_gdp, on='Code', how='left', suffixes=('_prj', ''))
# Drop the Region and Country fields from projects in favor of the more complete govtech
projects_gtmi = projects_gtmi.drop(columns=['Region_prj', 'Country_prj'])

# Examine the resulting data set
projects_gtmi

#%%

##############################################################################
### Regression of GTMI Data

# Correlation between the values
print("Correlation coefficient: " + str(govtech_gdp.GTMI.corr(govtech_gdp.GTEI)))

# Create a scatterplot of GTEI against GTMI with IncomeLevel as color
sns.relplot(x='GTMI', y='GTEI', data=govtech_gdp, kind='scatter', hue='IncomeLevel')
plt.show()

#%%

# Start by subsetting just the columns of interest
dfSub = govtech_gdp[['GTMI', 'GTEI', 'Population', 'GDP2021']].copy()
# Population is highly skewed. Let's remove dsome outliers.
dfPopulation = dfSub[dfSub['Population'] <= 300000]
dfSub['GDPMil'] = dfSub['GDP2021'] / 100000000

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Now plot the distribution of the remaining population values
ax[0].hist(dfPopulation['Population'], bins=10, color='lightblue', linewidth=1, edgecolor="black")
ax[0].set_xlabel('Population')
ax[0].set_title('Population Distribution')
ax[1].hist(dfSub['GDPMil'], bins=10, color='teal', linewidth=1, edgecolor="black")
ax[1].set_xlabel('GDP (in $100 million USD)')
ax[1].set_title('GDP 2021 Distribution')
plt.show()

#%%

# Calculate logarithmic values for Population and GDP
dfSub['logPopulation'] = np.log(dfSub['Population'])
dfSub['logGDP'] = np.log(dfSub['GDP2021'])

# Replot the distributions based on the logrithmic values
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(dfSub['logPopulation'], bins=10, color='lightblue', linewidth=1, edgecolor="black")
ax[0].set_xlabel('Log of Population')
ax[0].set_title('Population Distribution')
ax[1].hist(dfSub['logGDP'], bins=10, color='teal', linewidth=1, edgecolor="black")
ax[1].set_xlabel('Log of GDP (in USD)')
ax[1].set_title('GDP 2021 Distribution')
plt.show()

#%%

# Create a scatterplot of GTEI against GTMI with log of Population as color
sns.relplot(x='GTMI', y='GTEI', data=dfSub, kind='scatter', hue='logPopulation')
plt.show()

# Create a scatterplot of GTEI against GTMI with log of GDP as color
sns.relplot(x='GTMI', y='GTEI', data=dfSub, kind='scatter', hue='logGDP')
plt.show()

#%%

# Confirmed value of Logs
# Calculate Log of Popualtion and GDP and add as varaibles to the data frame
govtech_gdp['logPopulation'] = np.log(govtech_gdp['Population'])
govtech_gdp['logGDP'] = np.log(govtech_gdp['GDP2021'])

#%%

### Correlation of GTMI to TechSkills Training
### Start with correlation matrix

# We are breaking this into two matrices for simplicity in reading
columns = ['GTMI', 'GTEI', 'DS_Strategy_Program', 'FocusArea', 'DSProgram', 'DSProgramType']
subset = govtech_gdp[columns].copy()
subset.corr()

#%%

# And repeat with the remaining data
columns = ['GTMI', 'GTEI', 'DSProgramMandatory', 'DSProgramExternal', 'DSProgramPublished']
subset = govtech_gdp[columns].copy()
subset.corr()


#%%

### Regression of GTMI on TechSkills Training

# Import linregress from SciPy Stats library
from scipy.stats import linregress

# Call a simple regression function with the predictor followed by the dependent variable
model = linregress(govtech_gdp.GTEI, govtech_gdp.GTMI)

# Print results for interpretation
print(f'Intercept: {model.intercept:.4f}')
print(f'GTEI: {model.slope:.4f} ({model.stderr:.4f})')
print(f'R-Value: {model.rvalue:.4f}')
print(f'P-Value: {model.pvalue:.4f}')

#%%

# Scatterplot with regression
sns.lmplot(data=govtech_gdp, x='GTEI', y='GTMI')

#%%

# Import statsmodels
import statsmodels.formula.api as smf

#%%
# Create dummy variables out of the IncomeLevel categorical variable
dummies=pd.get_dummies(govtech_gdp['IncomeLevel'], prefix='IL', drop_first = True)
# Add the resulting dummies into the larger data set
govtech_gdp['IL_LM'] = pd.to_numeric(dummies['IL_LM'])
govtech_gdp['IL_UM'] = pd.to_numeric(dummies['IL_UM'])
govtech_gdp['IL_H'] = pd.to_numeric(dummies['IL_H'])

#%%

# Add in our independent variables to account for OVB, including the dummies created for IncomeLevel
model = smf.ols('GTMI ~ GTEI + logPopulation + logGDP + IL_LM + IL_UM + IL_H', data=govtech_gdp)
results = model.fit()
results.summary()

#%%

### This section not included in Notebook
# Control the output of the model
print(f'Intercept: {results.params.Intercept:.4f} ({results.bse.Intercept:.4f})')
print(f'GTEI: {results.params.GTEI:.4f} ({results.bse.GTEI:.4f})')
print(f'Log of Pop: {results.params.logPopulation:.4f} ({results.bse.logPopulation:.4f})')
print(f'Log of GDP: {results.params.logGDP:.4f} ({results.bse.logGDP:.4f})')

print(f'Income Level = LM: {results.params.IL_LM:.4f} ({results.bse.IL_LM:.4f})')
print(f'Income Level = UM: {results.params.IL_UM:.4f} ({results.bse.IL_UM:.4f})')
print(f'Income Level = H: {results.params.IL_H:.4f} ({results.bse.IL_H:.4f})')

print('------------------------------------')
print(f'Observations: {results.nobs:.0f}')
print(f'R-Squared: {results.rsquared:.4f}')
print(f'Deg. of F: {results.df_model:.0f}')

#results.rsquared_adj

#results.pvalues
#results.tvalues
#results.conf_int()   # default 95% confidence interval

#%%
