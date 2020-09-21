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
df5.head(20)
# %%
# Lets say that per bedroom 300 sqft is the average requirement
# So any valus which is very small or larger than 300 sqft is an anomaly and we can discard it
# we can find the anomalous data as:
df5[df5.total_sqft/df5.bhk<300].head()

# %%
df5.shape
# %%
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
# %%
df6.shape
# %%
df6.price_per_sqft.describe()
# %%
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
# %%
df7 = remove_pps_outliers(df6)
df7.shape
# %%
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location)&(df.bhk==2)]
    bhk3 = df[(df.location==location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
# %%
plot_scatter_chart(df7,'Rajaji Nagar')
# %%
plot_scatter_chart(df7,'Hebbal')
# %%
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
# %%
df8 = remove_bhk_outliers(df7)
df8.shape
# %%
plot_scatter_chart(df8,'Rajaji Nagar')
# %%
plot_scatter_chart(df8,'Hebbal')
# %%
import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel("Count")

# %%
# Lets explore bathroom feature now
df8.bath.unique()
# %%
# Checking out Homes with more than 10 Bathrooms
df8[df8.bath>10]
# %%
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
# %%
df8[df8.bath>df8.bhk+2]
# %%
# removing outliers in terms of number of bathrooms
df9 = df8[df8.bath<df8.bhk+2]
df9.shape
# %%
df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head()
# %%
