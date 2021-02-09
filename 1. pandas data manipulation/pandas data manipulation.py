
import numpy as np
import pandas as pd

# step 0 --------------------------------------------------
# function for conversion (more efficient)
def conv(x):
    if(type(x) == int):
        return(x * 1000000)
    else:
        return(x)
    
# read dataframe
energy = pd.read_excel('data/Energy Indicators.xls', skiprows = range(0,17), skipfooter = 38,
                usecols = 'C:F', names = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'],
                na_values = '...', converters={1:conv})

# modify Country column, they should be in this order
# remove digits using regex
energy['Country'] = energy['Country'].str.replace('\d+', '')
# remove parenthesis using regex
energy['Country'] = energy['Country'].str.replace(r"\(.*\)","").str.strip()
# rename 
energy['Country'].replace({"Republic of Korea": "South Korea", "United States of America": "United States",
                                "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                "China, Hong Kong Special Administrative Region": "Hong Kong"}, inplace=True)

# step 1 --------------------------------------------------
# read dataframe
gdp = pd.read_csv('data/world_bank.csv', skiprows = range(0,4))
# rename
gdp['Country Name'].replace({"Korea, Rep.": "South Korea", 
                             "Iran, Islamic Rep.": "Iran", 
                             "Hong Kong SAR, China": "Hong Kong"}, 
                            inplace=True)
gdp.rename(columns={'Country Name':'Country'}, inplace=True)

# step 2 --------------------------------------------------
# read dataframe
scimEn = pd.read_excel('data/scimagojr-3.xlsx')

# step 3 --------------------------------------------------
# merge scimEn, energy, gdp
df = pd.merge(scimEn, energy, on = 'Country')
#df = pd.merge(scimEn[:15], energy, on = 'Country')
df = pd.merge(df, gdp.iloc[:,np.r_[0,50:60]], on = 'Country')
old_shape = df.shape
df1 = df.iloc[0:15, :].copy() # to use it later 
new_shape = df1.shape
df1.set_index('Country', inplace = True)

# step 4 --------------------------------------------------
# entries lost
lost_entries = old_shape[0] - new_shape[0]

# step 5 --------------------------------------------------
# skip the Na values while finding the mean 
df1.iloc[:,-10:].mean(axis = 1, skipna = True).sort_values(ascending=False)
# df1.iloc[:,-10:].mean(axis = 1, skipna = True).nlargest(15)

# step 6 ------------------------------------------------
# mean Energy Supply per Capita
mean_per_capita = df1.loc[:, 'Energy Supply per Capita'].mean(axis = 0, skipna = True)

# step 7 ------------------------------------------------
# maximum % Renewable
max_renewable = df1['% Renewable'].idxmax(), df1.loc[df1['% Renewable'].idxmax(), '% Renewable']

# step 8 ------------------------------------------------
# pandas will handle 0/0 to nan, and n/0 to inf (not present in our case Citations >= Self-citations)
df1['ratio'] = df1['Self-citations'] / df1['Citations']
max_ratio = df1['ratio'].idxmax(), df1.loc[df1['ratio'].idxmax(), 'ratio']

# step 9 ------------------------------------------------
# compare with median
median_value = df1['% Renewable'].median(skipna = True) # get median
HighRenew = (df1['% Renewable'] >= median_value) * 1 # 1,0 compared to the median

# step 10 ------------------------------------------------
# define the dict
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}

# get datafame from the dict with col. Country, Continent
s = pd.Series(ContinentDict)
df_dict = s.reset_index()
df_dict.columns = ['Country', 'Continent']

# get datafame with col. Country, est_pop (estimated population)
pop = df1.copy()
pop['est_pop'] = pop['Energy Supply'] / pop['Energy Supply per Capita']
pop = pop.reset_index()
pop = pop[['Country', 'est_pop']]

# join both dataframes, to get col. Country, Continent, est_pop
gr = pd.merge(df_dict, pop, on = 'Country')
gr['est_pop'] = gr['est_pop'].apply(pd.to_numeric) 

# finally group to do the desired analysis, then concat them in one data frame
df_count = gr[['Continent', 'Country']].groupby('Continent').count().rename(columns={'Country': 'size'})
df_sum = gr[['Continent', 'est_pop']].groupby('Continent').sum().rename(columns={'est_pop': 'sum'})
df_mean = gr[['Continent', 'est_pop']].groupby('Continent').mean().rename(columns={'est_pop': 'mean'})
df_std_sample = gr[['Continent', 'est_pop']].groupby('Continent').std().rename(columns={'est_pop': 'std_sample'})
df_std_pop = gr[['Continent', 'est_pop']].groupby('Continent').std(ddof = 0).rename(columns={'est_pop': 'std_pop'})
df_analysis = pd.concat([df_count, df_sum, df_mean, df_std_sample, df_std_pop], axis=1)
