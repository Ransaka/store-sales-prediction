from copy import deepcopy

def remove_outliers(df,col):
  df = deepcopy(df)
  # for col in cols:
  mean = df[col].mean()
  std = df[col].std()
  
  lower_limit = mean - 3 * std
  upper_limit = mean + 3 * std
  
  n_inliers = df[col].between(lower_limit,upper_limit)
  return df[n_inliers]