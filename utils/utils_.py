from copy import deepcopy
from datetime import date, timedelta

def remove_outliers(df,col):
  df = deepcopy(df)
  # for col in cols:
  mean = df[col].mean()
  std = df[col].std()
  
  lower_limit = mean - 3 * std
  upper_limit = mean + 3 * std
  
  n_inliers = df[col].between(lower_limit,upper_limit)
  return df[n_inliers]

def all_sundays(year):
    dt = date(year, 1, 1)     
    dt += timedelta(days = 6 - dt.weekday())  
    while dt.year == year:
        yield dt
        dt += timedelta(days = 7)