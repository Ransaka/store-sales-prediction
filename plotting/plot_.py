import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

def plot_predictions(forecaster,region,cat):
    forecaster = deepcopy(forecaster)
    forecast = forecaster.results[region][cat]['forecast']
    error = forecaster.results[region][cat]['error']
    test = forecaster.results[region][cat]['test']
    test.set_index("ds",inplace=True)
    yhat_lower = forecast['yhat_lower']
    yhat_upper = forecast['yhat_upper']
    yhat = forecast['yhat']
    ds = forecast['ds']

    plt.figure(figsize=(15,7))
    plt.plot(ds,yhat,label='predictions')
    plt.fill_between(x=ds,y1=yhat_lower,y2=yhat_upper,color='gray', alpha=0.2)
    plt.plot(test['y'],label='True')
    plt.xticks(pd.date_range(min(ds),max(ds),periods=10))
    plt.legend(loc='upper right')
    plt.text(x=ds[0],y=max(yhat_upper),s=f"Mean Absulte Error: {error:.1f}",backgroundcolor='y')
    plt.title(f"Predictions vs. real data ({region}:{cat})".upper(),fontweight='semibold')
    plt.show()

def plot_region_contribution(dataset):
    df = deepcopy(dataset)
    
    region_map = map(lambda x:x.replace(" ","_").lower(),dataset.regions)
    region_revenue_map = {}
    for i,region in enumerate(region_map):
        regional_dataset = getattr(dataset,f"{region}_data")
        region_pct = regional_dataset.sum().sum()
        region_revenue_map[region] = region_pct

    plt.figure(figsize=(8,8))
    plt.pie(region_revenue_map.values(),labels=region_revenue_map.keys(),autopct='%.2f')
    plt.title(f"Region contibution to sales".upper(),fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_product_contribution(dataset):
    df = deepcopy(dataset)

    _,ax = plt.subplots(3,4,figsize=(15,2*8))
    axi = ax.flatten()

    regional_datasets = (getattr(dataset,f"region_{i+1}_data") for i in range(len(dataset.regions)) )
    for i,region_df in enumerate(regional_datasets):
        region_pct = region_df.sum()
        axi[i].pie(region_pct,labels=region_pct.index,autopct='%.2f')
        axi[i].set_title(f"Region {i+1}")

    plt.suptitle(f"Product sales distribution in each region".upper(),fontweight='bold',)
    plt.tight_layout()
    plt.show()

def plot_regional_dataset(data):
    df = deepcopy(data)
  
    _,ax = plt.subplots(nrows=df.shape[1],figsize=(15,8*df.shape[1]))

    for i,product in enumerate(df.columns):
        df[product].plot(ax=ax[i],color='#f25f0a')
        ax[i].set_title(f"{product} sales in given region",fontweight='semibold')
        ax[i].set_ylabel("demand")
    plt.show()