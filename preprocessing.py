import pandas as pd
from sklearn.impute import SimpleImputer

#import data
data = pd.read_csv("./insurance.csv")

# see the first 15 lines of data
# print(data.head(15))

# check how many values are missing (NaN) before we apply the methods below
count_nan = data.isnull().sum()  # the number of missing values for every column
print("#----------#\nMissing Values: \n",
      count_nan[count_nan > 0], "\n#----------#\n")

# option2 for filling NaN # reloading fresh dataset for option 2
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
# check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum()  # the number of missing values for every column
print("#----------#\nMissing Values: \n",
      count_nan[count_nan > 0], "\n#----------#\n")
