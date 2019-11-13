import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
cwd = os.getcwd()
path = cwd+"/income_estimator/data"
os.chdir(path)

# We first load the dataframe in which contains OAC groups for every postcode.
postcode_oac = pd.read_csv('2011 OAC May 2014 ONSPD Lookup.csv')
postcode_oac = postcode_oac[['PCDS', 'SUBGRP']]
# Load the data that maps OAC group to weekly household expenditure. This data has already been reformatted significantly in Excel to make it easier to work with.
postcode_oac.rename(columns= {'SUBGRP': 'OAC Code', 'PCDS': 'Postcode'}, inplace=True)

# Merge the dataframes
postcode_msoa = pd.read_csv('NSPCL_AUG19_UK_LU.csv', encoding='latin-1')


# This dataset also provides an OAC code however, the previous lookup appears to be more accurate (purely based on personal experience).
postcode_msoa = postcode_msoa[['pcds', 'lsoa11cd', 'msoa11cd']]

# Now merge data on the basis of postcode. First rename columns and drop Scottish and N.Irish records since they aren't present in the dataset we are merging with.

postcode_msoa.rename(columns={'pcds': 'Postcode', 'msoa11cd': 'MSOA', 'lsoa11cd': 'LSOA'}, inplace=True)

#Only want to keep MSOAs which start with E or W
postcode_msoa = postcode_msoa[postcode_msoa['MSOA'].str.startswith(('E', 'W'), na=False)]

oac_expenditure = pd.read_csv('OAC and Expenditure.csv')
oac_expenditure['OAC Code'] = oac_expenditure['OAC Code'].astype(str).str.lower()
postcode_oac_expenditure = pd.merge(postcode_oac, oac_expenditure, on='OAC Code')
postcode_oac_expenditure_msoa = pd.merge(postcode_msoa, postcode_oac_expenditure, on='Postcode')


# Now need to load data with estimated net income after housing costs for each MSOA and merge with the above.

msoa_income = pd.read_csv('MSOA_Income.csv')

msoa_income = msoa_income[['MSOA code', 'Net annual income before housing costs (£)']]
msoa_income.rename(columns={'MSOA code': 'MSOA', 'Net annual income before housing costs (£)': 'Net Income'}, inplace=True)
msoa_income['Net Income'] = msoa_income['Net Income'].str.replace(',', '')
msoa_income['Net Income'] = pd.to_numeric(msoa_income['Net Income'],errors='coerce')

all_data = pd.merge(postcode_oac_expenditure_msoa, msoa_income, on='MSOA')

#Need to produce some kind of dictionary, which tallies OAC Codes in each MSOA:LSOA pair, and then updates these
#values based on the net income assigned to that MSOA.
#First step is to get the net annual income in the dataframe. It might not be necessary to operate over a dictionary.
#but could instead have a column for each OAC subgroup and place the tally in there for a MSOA:LSOA pair
#and then for a unique MSOA, you tally total OAC groups (=unique postcodes), divide that by net annual income,
#multiply by each OAC in that thing and sum these totals for each OAC and place it into a dict (OAC: value)
#Nice! Now we want to nest this into another dictionary with MSOA as the key so LSOAs correspond to a MSOA.
#Keep the above as it is so we can access data by LSOA

MSOA_LSOA_dict = {k: {k1: v1.value_counts().to_dict() for k1, v1 in v.groupby('LSOA')['OAC Code']}
                                         for k, v in all_data.groupby('MSOA')}
#Good! Now to:
#1)Include Net Income in the data values or create a new dict for this purpose. It's gotta be a new dict
MSOA_Income_dict = pd.Series(all_data['Net Income'].values,index=all_data['MSOA']).to_dict()
#2)For a MSOA, tally all of the OAC inside so E0200001 2d3 is 145+280+3717 etc
#3)Divide each of these by total OAC count within that MSOA and multiply by Net Income for the MSOA
#4)Create a dict for OAC and their total value across all MSOA
#5)Calculate these all as a proportion of total so they have weights
totals = defaultdict(int)
total_total = 0

#For a MSOA, get the frequencies of OACs present within it
for key in MSOA_LSOA_dict.keys():
    result = defaultdict(int)
    total = 0
    vals = MSOA_LSOA_dict[key].values()

    for d in vals:
        for k, v in d.items():
            result[k] += v
            total += v

    for k in result:
        result[k] /= total
        result[k] *= MSOA_Income_dict[key] #multiply by income

        for k, v in result.items():
            totals[k] += v
            total_total += v

for k in totals:
    totals[k] /= total_total

for key in MSOA_LSOA_dict.keys():
    vals = MSOA_LSOA_dict[key].values()
    for d in vals:
        for k, v in d.items():
            d[k] = 1

#Let's work with a dataframe again
all_data['Weight'] = all_data['OAC Code'].map(totals)
cols = all_data.columns.tolist()
cols[1], cols[2] = cols[2], cols[1]
all_data=all_data[cols]

all_data['Total Weekly Expenditure'] = pd.to_numeric(all_data['Total Weekly Expenditure'])
all_data['Average people per household'] = pd.to_numeric(all_data['Average people per household'])


res = (all_data.groupby(['MSOA', 'LSOA'])
          .agg({'Weight': 'mean',
                'Total Weekly Expenditure': 'mean',
                'Average people per household': 'mean',
                'Postcode': ','.join,
                'Net Income': 'first'}))

res['MSOA Income'] = res['Net Income']
s = res.groupby(level='MSOA')
res['Net Income'] = s['Net Income'].transform('sum')*res.Weight/s.Weight.transform('sum')
res.drop(columns=['Weight'], axis=1, inplace=True)

res.reset_index(inplace=True)

res2 = res['MSOA'].values
res2 = np.unique(res2)
new_income = []

for MSOA in res2:
    this = res.loc[res['MSOA'] == MSOA]
    MSOA_Income = this['MSOA Income'].iloc[0]
    data = (this['Net Income'].values).reshape(-1,1)
    min_max_scaler = MinMaxScaler(feature_range=(MSOA_Income*1.4, MSOA_Income*1.9))
    new_income.append(min_max_scaler.fit_transform(data))

new_income = np.vstack(new_income).ravel().tolist()

res['Net Income'] = new_income
res['Postcode'] = res['Postcode'].str.split(',')
res = res.explode('Postcode') #Now just to place column back there
#Merge with Expenditure Data
#postcode_oac_expenditure = postcode_oac_expenditure[['Postcode', 'Total Weekly Expenditure', 'Average people per household']]
#res = pd.merge(res, postcode_oac_expenditure, on = 'Postcode')

res['Total Weekly Expenditure'] = pd.to_numeric(res['Total Weekly Expenditure'],errors='coerce')
res['Weekly Disposable'] = ((res['Net Income']/52) - res['Total Weekly Expenditure'])/res['Average people per household']
res.drop(columns=['Net Income', 'Total Weekly Expenditure'])
res = (res.groupby(['Weekly Disposable']).agg({'Postcode': ','.join}))

income_split = pd.read_csv('Income split by Age and Gender.csv', index_col = 0)

bins = np.linspace(25,75,11)

def getIncome(age, gender, postcode):

    if (age < 20):
        age_band = 0
    elif (age >= 75):
        age_band = 12
    else:
        index = np.digitize(age, bins)
        age_band = index+1

    proportion = income_split.iloc[age_band][gender]
    disposable = res.loc[res['Postcode'].str.contains(postcode)].index[0]
    disposable = disposable * proportion
    disposable = disposable * 52

    return disposable

os.chdir(cwd)
