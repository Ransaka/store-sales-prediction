from utils.utils_ import remove_outliers
from pandas import pivot_table,DatetimeIndex,DataFrame
from tqdm import tqdm

class ShopDataset():
    "WonderCo dataset class"
    def __init__(self,data,region_column,product_col,date_column,target_col) -> None:
        """
        data: pandas dataframe including all information of invoice sales/return 
        region_column: this is the column name related to the regions in the dataset
        product_col: this is the column name related to the product categpry in the dataset
        date_column: time stamp column in the dataset
        target_col: the column name, that we are going to forecast
        """
        assert isinstance(data,DataFrame),"data must be a pandas dataframe"
        assert isinstance(region_column,str),"`region_columns` must be a str type"
        assert isinstance(product_col,str),"`product_col` must be a str type"
        assert isinstance(date_column,str),"`date_column` must be a str type"

        self.data = remove_outliers(data,target_col)
        self.target_col = target_col
        self.region_col = region_column
        self.product_col = product_col
        self.date_column = date_column
        self.regions = data[region_column].unique().tolist()
        self.products = data[product_col].unique().tolist()
        self.data[date_column] = DatetimeIndex(self.data[date_column])

    def define_regional_datasets(self):
        """This will define a datasets for each regions in the dataset. User can access the individual region data using `dataset_instance.regoin_{i}_data`"""
        setattr(self,'dataset_names',[])
        for region in tqdm(self.regions):
            d_ = self.data.query(f"{self.region_col}=='{region}'")
            regional_data = pivot_table(
                data = d_,
                index=self.date_column,
                values=self.target_col,
                columns=self.product_col,
                aggfunc=sum,
                fill_value=0
            )
            regional_data = regional_data.asfreq("D").fillna(0) #define daily frequency for dataset and all missing with 0. Assumption: Missing means no sales for that day.
            region_name = region.replace(" ","_").lower()
            setattr(self,f"{region_name}_data",regional_data)
            self.dataset_names.append(f"{region_name}_data")