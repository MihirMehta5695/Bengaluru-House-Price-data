# %%
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (20, 10)

# %%
file_name = 'datasets_20710_26737_Bengaluru_House_Data.csv'
# df stands for data_frame
df1 = pd.read_csv('../data/'+file_name)
# prints the first five rows of the datasets
df1.head()

# %%
df1.shape  # output-> (number_of_rows,number_of_columns)

# %%
df1.groupby('area_type')['area_type'].agg('count')

# %%
df2 = df1.drop(['area_type','society','balcony','availability'],axis="columns")
df2.head()
# %%
# isnull method will return the rows with null value
df2.isnull().sum()
# %%
# dropna method drops all the values with null or na (Not Applicable)
df3 = df2.dropna()
df3.isnull().sum()
df3.shape
# %%
# Selecting the column and fetching all unique values from it
df3['size'].unique()
# %%
# We will create a new column into df3 dataset as follows:
df3['bhk'] = df3['size'].apply(lambda  x: int(x.split(' ')[0]))

# %%
df3.head()
# %%
df3['bhk'].unique()
# %%
df3[df3.bhk>20]
# %%
df3.total_sqft.unique()
# %%
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
# %%
# ~ is a negation operation
df3[~df3["total_sqft"].apply(is_float)].head(10)
# Helps in checking unstructure/unclean data
# %%
def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

# %%
# Testing our above function
convert_sqft_to_num('2166')
convert_sqft_to_num('2000 - 3000')
convert_sqft_to_num('34.46Sq.meter')
# %%
# data_frame.copy() will create a deep copy
df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)
df4.head()
# %%
df4.loc[30]

# 1_Data_Cleaning.py code ends here
# %%
df5 = df4.copy()
df5['price_per_sqft']=(df5['price']*100000)/df5['total_sqft']
df5.head()
# %%
df5.location.unique()
len(df5.location.unique())
# %%
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats
# %%
len(location_stats[location_stats<=10])

# %%
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10
# %%
len(df5.location.unique())
# %%
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# %%
len(df5.location.unique())
# %%
