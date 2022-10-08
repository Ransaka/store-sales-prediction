from data.dataset import ShopDataset
from tqdm import tqdm
from prophet import Prophet
from sklearn import metrics
from types import NoneType
from pandas import DataFrame

class Forecaster():
    """The ABC forecasting model"""
    def __init__(self,dataset,holidays,test_size=0.2) -> None:
        assert isinstance(dataset,ShopDataset)
        assert 0 < test_size < 1,"test size should be in range 0 and 1"
        """
        dataset: This should be a ShopDataset and should be call .define_regional_datasets() method prior adding to this.
        holidays: holidays dataframe. This should be a prophet compatible dataframe. Info: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events
        test_size: this should be a float value between 0 and 1. Depending on it's value 
        each regional dataset will be splitted into train and test. Test dataset will use for calculate the model performance.
        """
        self.dataset = dataset 
        self.holidays = holidays
        self.test_size = test_size

    def fit(
        self,
        limit=None,
        include_history=False,
        add_country_holidays=False,
        weekly_seasonality='auto',
        yearly_seasonality='auto'
        ):
        """
        limit None or integer: if you want to limit models, you may set this to desired no. of models
        include_history: prophet compatible parameter. See Prophet parameters for more info.
        weekly_seasonality: prophet compatible parameter. See Prophet parameters for more info.
        yearly_seasonality: prophet compatible parameter. See Prophet parameters for more info.
        add_country_holidays:If you want to add country holydays, you may add holiday dataframe. This sould be compatible with `Prophet`
        """
        assert isinstance(limit,(NoneType,int))
        assert isinstance(include_history,bool)
        assert isinstance(add_country_holidays,bool)
        assert isinstance(weekly_seasonality,(str,DataFrame,bool))
        assert isinstance(yearly_seasonality,(str,DataFrame,bool))

        self.add_country_holidays = add_country_holidays
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        dataset_names = self.dataset.dataset_names
        if limit:dataset_names = self.dataset.dataset_names[:limit]
        self.results = {region:{product:{"forecast":None,"model":None,"error":None,'test':None} for product in self.dataset.products} for region in dataset_names}
        self.meta_info = {dataset_name:[x for x in getattr(self.dataset,dataset_name)] for dataset_name in dataset_names}#Since all products are not avaialbe in all regions, keep region and product mapping
        for dataset_name in tqdm(self.meta_info):
            for category in self.meta_info[dataset_name]:
                forecast,model,error,test = self.run_prophet(dataset_name=dataset_name,category=category,include_history=include_history)
                self.results[dataset_name][category]['forecast'] = forecast
                self.results[dataset_name][category]['model'] = model
                self.results[dataset_name][category]['error'] = error
                self.results[dataset_name][category]['test'] = test
                
    def get_ts(self,dataset_name,category):
        _dataset = getattr(self.dataset,dataset_name)
        ts = _dataset[[category]]
        ts.reset_index(inplace=True)
        ts.columns = ['ds','y']
        train_idx = int(len(ts) * (1-self.test_size))
        train = ts[:train_idx]
        test = ts[train_idx:]
        return train,test

    def run_prophet(self,dataset_name,category,include_history=False):
        train,test = self.get_ts(dataset_name,category)
        periods = len(test)
        model = Prophet(holidays=self.holidays,yearly_seasonality=self.yearly_seasonality,weekly_seasonality=self.weekly_seasonality)
        if self.add_country_holidays:model.add_country_holidays(country_name=self.add_country_holidays)
        model.fit(train)
        forecast = model.make_future_dataframe(periods=periods, include_history=include_history)
        forecast = model.predict(forecast)
        forecast['yhat'] = forecast['yhat'].apply(lambda x:0 if x<0 else x)#cause demand cant be negetive
        error = metrics.mean_absolute_error(test['y'],forecast['yhat'])
        return forecast,model,error,test