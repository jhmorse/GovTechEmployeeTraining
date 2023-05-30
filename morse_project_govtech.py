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
## Load GovTech data

# Path where files are located
path = 'C:/Users/jomors/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/Data/'
#path = 'C:/Users/johnh/OneDrive/_JHU/AS.470.708 Open Data in Python/Project/Data/'

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
## Update column data types in GovTech data
# Several of the columns get read in as objects instead of numeric.
# Need to convert these to numeric for evaluation.
govtech['DSProgram'] = pd.to_numeric(govtech.DSProgram)
govtech['DSProgramType'] = pd.to_numeric(govtech.DSProgramType)
govtech['DSProgramMandatory'] = pd.to_numeric(govtech.DSProgramMandatory)
govtech['DSProgramExternal'] = pd.to_numeric(govtech.DSProgramExternal)
govtech['DSProgramPublished'] = pd.to_numeric(govtech.DSProgramPublished)
govtech.info()

#%%

## Load GovTech Projects Data
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

## Load GDP Data
#filename = '../Data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4770391.csv'
filename = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4770391.csv'
gdp_raw = pd.read_csv(path + filename, header=2)

# Select columns of interest
gdp = gdp_raw[['Country Name', 'Country Code', '2021']]
# Rename columns
gdp = gdp.rename(columns={'Country Name':'Country', 'Country Code':'Code', '2021':'GDP2021'})

gdp.head()

#%%
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
## Convert IncomeLevel column to a categorical varaibles

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
### Prep Projects data

## Start with examination of Outcome scores

# Distribution of ICROutcome values
category_order = ['HU', 'U', 'MU', 'MS', 'S', 'HS']
sns.catplot(x='ICROutcome', data=projects, kind='count', order=category_order, color='teal')
plt.show()


# Distribution of IEGOutcome values
sns.catplot(x='IEGOutcome', data=projects, kind='count', order=category_order, color='navy')
plt.show()

#%%
### Convert Outcome variables to bivariate
## We will use logistic regression for the outcome variables,
## so we need to convert to binary variables.

# Create binary values for ICROutcome and IEGOutcome
projects['ICROutcomeB'] = np.where(projects['ICROutcome'].isin(['HS', 'S', 'MS']), 1, 0)
projects['IEGOutcomeB'] = np.where(projects['IEGOutcome'].isin(['HS', 'S', 'MS']), 1, 0)

# Check the totals for each binary variable
print(np.sum(projects.ICROutcomeB))
print(np.sum(projects.IEGOutcomeB))

# Display the first few columns of the data set
projects.head(10)

#%%
### Add GTMI data to Projects data
## Now we can join project data to GTMI data to use the Tech Enablment variables
## in the analysis of the project outcome status

# Create a set of Country names and Country Codes from the govtech data.
country_lookup = govtech[['Country', 'Code']]

# Join this to the projects data based on the Country name
projectscd = projects.merge(country_lookup, on='Country', how='left', suffixes=('_prj', '_gt'))

# preview the data
projectscd.head(10)

#%%

## Some countries did not get picked up. 
## We need to manually create a lookup data set.

# Which countries did not get picked up?
projectscd[pd.isna(projectscd['Code'])]['Country'].unique()

#%%
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

# Get some basic stats on the project data and drop Unknown codes
print('Total projects: ' + str(len(projectscd)))
print('Projects with unknown countries: ' + str(np.sum(projectscd.Code == 'UNK')))

# Drop the projects with Unknown country codes
projectscd = projectscd[projectscd['Code'] != 'UNK']
print('Remaining projects: ' + str(len(projectscd)))

#%%
## Final merge of porjects and GTMI data
# Merge the projects and govtech data
projects_gtmi = projectscd.merge(govtech_gdp, on='Code', how='left', suffixes=('_prj', ''))
# Drop the Region and Country fields from projects in favor of the more complete govtech
projects_gtmi = projects_gtmi.drop(columns=['Region_prj', 'Country_prj'])

# Examine the resulting data set
projects_gtmi

#%%
### Examine GTEI and GTMI data in scatterplot

#################################################################################
### Controlling for OVB

## We are going to consider three variables as control variables.
## This section examines how those variables are distributed across
## The GovTech data.

# Create a scatterplot of GTEI against GTMI with IncomeLevel as color
ILScatter = sns.relplot(x='GTMI', y='GTEI', data=govtech_gdp, kind='scatter', hue='IncomeLevel')
ILScatter.set(title='Income Level Across Tech Maturity and Tech Enablement Scores')

#%%

## Now let's look at distribution of Population and GDP data
# Start by subsetting just the columns of interest
dfSub = govtech_gdp[['GTMI', 'GTEI', 'Population', 'GDP2021']].copy()
# Population is highly skewed. Let's remove dsome outliers.
dfPopulation = dfSub[dfSub['Population'] <= 300000]
# And for simplicity of understnaidng the data, let's convert GDP to 100s of millions
dfSub['GDPMil'] = dfSub['GDP2021'] / 100000000

# Now plot the distributions of both values
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(dfPopulation['Population'], bins=10, color='lightblue', linewidth=1, edgecolor="black")
ax[0].set_xlabel('Population')
ax[0].set_title('Population Distribution')
ax[1].hist(dfSub['GDPMil'], bins=10, color='teal', linewidth=1, edgecolor="black")
ax[1].set_xlabel('GDP (in $100 million USD)')
ax[1].set_title('GDP 2021 Distribution')
plt.show()  

#%% 
## Since the distributions are skewed, let's repeat with Log values
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

## Create scatter plots with Log of Population and Log of GDP
# Create a scatterplot of GTEI against GTMI with log of Population as color
logPopScatter = sns.relplot(x='GTMI', y='GTEI', data=dfSub, kind='scatter', hue='logPopulation')
logPopScatter.set(title='Log of Population Across Tech Maturity and Tech Enablement Scores')
#plt.show()

# Create a scatterplot of GTEI against GTMI with log of GDP as color
logGDPScatter = sns.relplot(x='GTMI', y='GTEI', data=dfSub, kind='scatter', hue='logGDP')
logGDPScatter.set(title='Log of GDP Across Tech Maturity and Tech Enablement Scores')

#%% 
## Now add these Log values to the data
# Calculate Log of Popualtion and GDP and add as varaibles to the data frame
govtech_gdp['logPopulation'] = np.log(govtech_gdp['Population'])
govtech_gdp['logGDP'] = np.log(govtech_gdp['GDP2021'])
#%%
### Correlation of GTMI to TechSkills Training - Start with correlation matrix

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
print(f'R-Squared: {model.rvalue**2:.4f}')
print(f'P-Value: {model.pvalue:.4f}')
print(f'Observations: {govtech_gdp.shape[0]}')

#%%
# Scatterplot with regression
simpleLin = sns.lmplot(data=govtech_gdp, x='GTEI', y='GTMI')
simpleLin.set(title='Linear Regression of GTEI on GTMI')


#%%

# Import statsmodels for more complete linear regression
import statsmodels.formula.api as smf

#%%
# Create dummy variables out of the IncomeLevel categorical variable
dummies=pd.get_dummies(govtech_gdp['IncomeLevel'], prefix='IL', drop_first = True)
# Add the resulting dummies into the larger data set
govtech_gdp['IL_LM'] = pd.to_numeric(dummies['IL_LM'])
govtech_gdp['IL_UM'] = pd.to_numeric(dummies['IL_UM'])
govtech_gdp['IL_H'] = pd.to_numeric(dummies['IL_H'])

#%%

# Regression with our independent variables to account for OVB, including the dummies created for IncomeLevel
model = smf.ols('GTMI ~ GTEI + logPopulation + logGDP + IL_LM + IL_UM + IL_H', data=govtech_gdp)
results = model.fit()
results.summary()

#%%

### This section not included in Notebook - Control the output of the model
print(f'Intercept: {results.params.Intercept:.4f} ({results.bse.Intercept:.4f})')
print(f'GTEI: {results.params.GTEI:.4f} ({results.bse.GTEI:.4f})')
print(f'Log of Pop: {results.params.logPopulation:.4f} ({results.bse.logPopulation:.4f})')
print(f'Log of GDP: {results.params.logGDP:.4f} ({results.bse.logGDP:.4f})')

print(f'Income Level = LM: {results.params[1]:.4f} ({results.bse[1]:.4f})')
print(f'Income Level = UM: {results.params[2]:.4f} ({results.bse[2]:.4f})')
print(f'Income Level = H: {results.params[3]:.4f} ({results.bse[3]:.4f})')

print('------------------------------------')
print(f'Observations: {results.nobs:.0f}')
print(f'R-Squared: {results.rsquared:.4f}')
print(f'Deg. of F: {results.df_model:.0f}')



#%%
## Next, extend the regression to cover all independent variables
# Build complete regression for GTMI
model = smf.ols('GTMI ~ DS_Strategy_Program + FocusArea + DSProgram + DSProgramType + \
                DSProgramMandatory + DSProgramExternal + DSProgramPublished +\
                logPopulation + logGDP + IL_LM + IL_UM + IL_H',    # Control variables for OVB
                data=govtech_gdp)
results = model.fit()
results.summary()



# %%
## Repeat but regression on GTEI as the dependent variable
# Build complete regression for GTEI
model = smf.ols('GTEI ~ DS_Strategy_Program + FocusArea + DSProgram + DSProgramType + \
                DSProgramMandatory + DSProgramExternal + DSProgramPublished +\
                logPopulation + logGDP + IL_LM + IL_UM + IL_H',    # Control variables for OVB
                data=govtech_gdp)
results = model.fit()
results.summary()

# %%
## Add Log values to Porjects data

# Add Log values of Population and GDP to the data set
projects_gtmi['logPopulation'] = np.log(projects_gtmi['Population'])
projects_gtmi['logGDP'] = np.log(projects_gtmi['GDP2021'])

# Convert the IncomeLevel column to categorical
projects_gtmi['IncomeLevel'] = projects_gtmi['IncomeLevel'].astype(cat_dtype)
# Create dummy variables out of the IncomeLevel categorical variable
dummies=pd.get_dummies(projects_gtmi['IncomeLevel'], prefix='IL', drop_first = True)
# Add the resulting dummies into the larger data set
projects_gtmi['IL_LM'] = dummies['IL_LM']
projects_gtmi['IL_UM'] = dummies['IL_UM']
projects_gtmi['IL_H'] = dummies['IL_H']

# Review the Projects data
projects_gtmi.head()

# %%
## Regression of Project Outcomes - Start with Generalized Linear Model from statsmodels

# Logistic regression on ICROutcome binary variable using statsmodels
model_icr = smf.glm('ICROutcomeB ~ DS_Strategy_Program + FocusArea + DSProgram + DSProgramType + \
                    DSProgramMandatory + DSProgramExternal + DSProgramPublished +\
                    logPopulation + logGDP + IL_LM + IL_UM + IL_H',    # Control variables for OVB
                    data=projects_gtmi)
results_icr = model_icr.fit()
results_icr.summary()


# %%
# Logistic regression on IEGOutcome binary variable using statsmodels
model_ieg = smf.glm('IEGOutcomeB ~ DS_Strategy_Program + FocusArea + DSProgram + DSProgramType + \
                    DSProgramMandatory + DSProgramExternal + DSProgramPublished +\
                    logPopulation + logGDP + IL_LM + IL_UM + IL_H',    # Control variables for OVB
                    data=projects_gtmi)
results_ieg = model_ieg.fit()
results_ieg.summary()


# %%
## As an alternative, bring in LogisticRegression from the SciKit-Learn library and repeat process.

# import sklearn libraries

from sklearn.linear_model import LogisticRegression

# %%
## Build numpy arrays out of the data frame variables

# Pull out the independent variables from the data frame
# and turn them into a numpy 2D array
columns = ['DS_Strategy_Program','FocusArea', 'DSProgram', 'DSProgramType', 'DSProgramMandatory', 'DSProgramExternal']
controls = ['logPopulation', 'IL_LM', 'IL_UM', 'IL_H']

# Establish our array of predictors
predictors = np.array(projects_gtmi[columns + controls])
# Turn the dependent variable into a numpy array as well.

## We start with the ICR Outcome variable.
outcome_icr = np.array(projects_gtmi['ICROutcomeB'])
# Reshape the outcome varaible into a 2D array
outcome_icr = outcome_icr.reshape(-1,1)

# %%
## sklearn logistic regression
# Create base LogisticRegression function
logit_icr = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
# Generate the model based on the arrays created above.
logit_icr.fit(X=predictors, y=outcome_icr)

#%%
## Calculate Standard Errors through a covariance matrix.
## SciKit-Learn does not support this directly.

# Calculate matrix of predicted class probabilities.
predProbs_icr = logit_icr.predict_proba(predictors)

# Design matrix -- add column of 1's at the beginning of yoru predictors matrix
X_design = np.hstack([np.ones((predictors.shape[0], 1)), predictors])
# Initiate matrix of 0s, fill diagonal with each predicted observation's variance
V = np.diagflat(np.product(predProbs_icr, axis=1))

# Covariance matrix
covLogit = np.linalg.inv(np.dot(np.dot(predictors.T, V), predictors).astype(np.float64))

# Standard errors
coef_se = np.sqrt(np.diag(covLogit))

#%%
## Print out Intercept and coefficients for refernece
print(f'Intercept: {logit_icr.intercept_}')
print(pd.DataFrame({'coeff': logit_icr.coef_[0], 'se': coef_se},index=columns + controls))

#%%
## Repeat the process for IEGOutcomeB

# Turn the dependent variable into a numpy array.
# Note that we will use the saem predictors as before.
outcome_ieg = np.array(projects_gtmi['IEGOutcomeB'])
# Reshape the outcome varaible into a 2D array
outcome_ieg = outcome_ieg.reshape(-1,1)

# Create base LogisticRegression function
logit_ieg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
# Generate the model based on the arrays created above.
logit_ieg.fit(X=predictors, y=outcome_ieg)

#%%
# Calculate Standard Errors through a covariance matrix.
## SciKit-Learn does not support this directly.

# Calculate matrix of predicted class probabilities.
predProbs_ieg = logit_ieg.predict_proba(predictors)

# Design matrix -- add column of 1's at the beginning of yoru predictors matrix
X_design = np.hstack([np.ones((predictors.shape[0], 1)), predictors])
# Initiate matrix of 0s, fill diagonal with each predicted observation's variance
V = np.diagflat(np.product(predProbs_icr, axis=1))

# Covariance matrix
covLogit = np.linalg.inv(np.dot(np.dot(predictors.T, V), predictors).astype(np.float64))

# Standard errors
coef_se = np.sqrt(np.diag(covLogit))
#%%
## Print out Intercept and coefficients for refernece
print(f'Intercept: {logit_ieg.intercept_}')
print(pd.DataFrame({'coeff': logit_ieg.coef_[0], 'se': coef_se},index=columns + controls))

#%%
### Prediction with Logistic Models

## For the statsmodels GLM models, we will simply compare the predicted 
## number of 1s to the actual number of 1s provided in the origninal data.

# For ICROutcome, what percentage are actaully 1
act_icr = np.array(projects_gtmi['ICROutcomeB'])
act_icr_pct = sum(act_icr) / np.shape(act_icr)[0]
# Based on a threshold > 0.5, what percentage were predicted to be 1
pred_icr = results_icr.predict()
pred_icr_pct = sum(pred_icr > 0.5) / np.shape(pred_icr)[0]
# Print out the results
print('ICR Outcome Results')
print(f'Actual Percent of Satisfactory Results: {act_icr_pct:.4f}')
print(f'Predicted Percent of Satisfactory Results: {pred_icr_pct:.4f}')

print('\n')
# For IEGOutcome, what percentage are actaully 1
act_ieg = np.array(projects_gtmi['IEGOutcomeB'])
act_ieg_pct = sum(act_ieg) / np.shape(act_ieg)[0]
# Based on a threshold > 0.5, what percentage were predicted to be 1
pred_ieg = results_ieg.predict()
pred_ieg_pct = sum(pred_ieg > 0.5) / np.shape(pred_ieg)[0]
# Print out the results
print('IEG Outcome Results')
print(f'Actual Percent of Satisfactory Results: {act_ieg_pct:.4f}')
print(f'Predicted Percent of Satisfactory Results: {pred_ieg_pct:.4f}')


# %%

## Now use a Classification Report from SciKit-Learn
from sklearn.metrics import classification_report

# Print Classification report for ICR Outcome
print('Classification Report: ICR Outcome from Logit Model')
print(classification_report(outcome_icr, logit_icr.predict(predictors)))

# Print a newline
print('\n')

# Print Classification report for ICR Outcome
print('Classification Report: IEG Outcome from Logit Model')
print(classification_report(outcome_ieg, logit_ieg.predict(predictors)))


# %%
