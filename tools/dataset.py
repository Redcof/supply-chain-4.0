import logging
import pathlib

import pandas as pd

from tools.preprocess import preprocess

root_dir = pathlib.Path(r"./datasets")
logger = logging.getLogger(__name__)


def get_dataset(dataset_name, is_extra_feature_enabled=False, ablation_limit=None):
    logger.info(f"Creating dataset for :{dataset_name}")
    if dataset_name == "food_demand":
        df, target, timeseries_col, dataset_name = food_demand_dataset(
            is_extra_feature_enabled=is_extra_feature_enabled)
    elif dataset_name == "product_demand":
        df, target, timeseries_col, dataset_name = forecasts_for_product_demand_dataset(
            is_extra_feature_enabled=is_extra_feature_enabled)
    elif dataset_name == "livestock_meat_import":
        df, target, timeseries_col, dataset_name = livestock_meat_dataset(
            is_extra_feature_enabled=is_extra_feature_enabled)
    elif dataset_name == "online_retail":
        df, target, timeseries_col, dataset_name = online_retail_dataset(
            is_extra_feature_enabled=is_extra_feature_enabled)
    elif dataset_name == "online_retail_2":
        df, target, timeseries_col, dataset_name = online_retail_2_dataset(
            is_extra_feature_enabled=is_extra_feature_enabled)
    else:
        raise ValueError("Invalid dataset name")
    if ablation_limit != -1 and isinstance(ablation_limit, int):
        df = df.head(ablation_limit)
    # remove columns with constant values
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df, target, timeseries_col, dataset_name


def food_demand_dataset(is_extra_feature_enabled=False):
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_csv(root_dir / "food_demand_dataset" / "train.csv", index_col='id')
    df.dropna(inplace=True)
    # fix dtypes
    df['center_id'] = df['center_id'].astype('category')
    df['meal_id'] = df['meal_id'].astype('category')
    df['emailer_for_promotion'] = df['emailer_for_promotion'].astype('category')
    df['homepage_featured'] = df['homepage_featured'].astype('category')
    # sort
    df['week'] = df['week'].astype('int32')
    df.sort_values(by=['week'], inplace=True)
    df['week'] = df['week'].astype('category')
    # preprocess
    df = preprocess(df, "num_orders", is_extra_feature_enabled=is_extra_feature_enabled)
    return df, "num_orders", None, "food_demand"


def forecasts_for_product_demand_dataset(is_extra_feature_enabled=False):
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_csv(root_dir / "forecast_product_demand" / "forecasts_for_product_demand.csv")
    df.dropna(inplace=True)
    # treat Order_Demand
    df['Order_Demand'] = df['Order_Demand'].str.replace('(', "")
    df['Order_Demand'] = df['Order_Demand'].str.replace(')', "")
    df['Order_Demand'] = df['Order_Demand'].astype('int64')
    # fix dtypes
    df['Product_Code'] = df['Product_Code'].astype('category')
    df['Warehouse'] = df['Warehouse'].astype('category')
    df['Product_Category'] = df['Product_Category'].astype('category')
    # sort by datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date'], inplace=True)
    # preprocess
    df = preprocess(df, "Order_Demand", "Date", is_extra_feature_enabled=is_extra_feature_enabled)
    return df, "Order_Demand", "Date", "product_demand"


def livestock_meat_dataset(is_extra_feature_enabled=False):
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_csv(root_dir / "livestock_meat" / "livestock_meat_imports.csv")
    df.dropna(inplace=True)
    # dtypes
    df['HS_CODE'] = df['HS_CODE'].astype('int64').astype('category')
    df['GEOGRAPHY_CODE'] = df['GEOGRAPHY_CODE'].astype('int64').astype('category')
    df['TIMEPERIOD_ID'] = df['TIMEPERIOD_ID'].astype('int64').astype('category')
    df['GEOGRAPHY_DESC'] = df['GEOGRAPHY_DESC'].astype('category')
    df['YEAR_ID'] = df['YEAR_ID'].astype('int64').astype('category')
    df['UNIT_DESC'] = df['UNIT_DESC'].astype('category')
    # sort
    df.sort_values(by=['YEAR_ID', 'TIMEPERIOD_ID'], inplace=True)
    # drop
    df.drop(columns=["SOURCE_ID", "COMMODITY_DESC", "ATTRIBUTE_DESC"], inplace=True)
    # TODO Convert 'COMMODITY_DESC' feature to categories
    # preprocess
    df = preprocess(df, "AMOUNT", is_extra_feature_enabled=is_extra_feature_enabled)
    return df, "AMOUNT", None, "livestock_meat_import"


def online_retail_dataset(is_extra_feature_enabled=False):
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail" / "online_retail.xlsx")
    df.dropna(inplace=True)
    # fix dtypes
    df['CustomerID'] = df['CustomerID'].astype('int64').astype('category')
    df['Country'] = df['Country'].astype('category')
    df['StockCode'] = df['StockCode'].astype('string')
    # datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.sort_values(by=['InvoiceDate'], inplace=True)
    invoice_date = df.InvoiceDate.copy()
    # preprocess
    df = preprocess(df, "Quantity", "InvoiceDate", is_extra_feature_enabled=is_extra_feature_enabled)
    # drop unusable columns
    df.drop(columns=['InvoiceNo', 'Description'], inplace=True)
    # TODO Convert 'Description' feature to categories
    return df, "Quantity", "InvoiceDate", "online_retail"


def online_retail_2_dataset(is_extra_feature_enabled=False):
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail_2" / "online_retail_2.xlsx",
                       sheet_name=['Year 2009-2010', 'Year 2010-2011'])

    df = pd.concat([df['Year 2009-2010'], df['Year 2010-2011']])
    df.dropna(inplace=True)
    # fix dtypes
    df['Customer ID'] = df['Customer ID'].astype('int64').astype('category')
    df['Country'] = df['Country'].astype('category')
    df['StockCode'] = df['StockCode'].astype('string')
    # datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.sort_values(by=['InvoiceDate'], inplace=True)
    # preprocess
    df = preprocess(df, "Quantity", "InvoiceDate", is_extra_feature_enabled=is_extra_feature_enabled)
    # drop unusable columns
    df.drop(columns=['Invoice', 'Description'], inplace=True)
    # TODO Convert 'Description' feature to categories

    return df, "Quantity", "InvoiceDate", "online_retail_2"
