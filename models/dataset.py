import pathlib

import pandas as pd

from models.preprocess import preprocess

root_dir = pathlib.Path(r"../datasets")


def food_demand_dataset():
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
    df = preprocess(df, "num_orders")
    return df, "num_orders", pd.Series(list(range(len(df))))


def forecasts_for_product_demand_dataset():
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
    datetime = df['Date'].copy()
    # preprocess
    df = preprocess(df, "Order_Demand", "Date")
    # drop datetime
    df.drop(columns=['Date'], inplace=True)
    return df, "Order_Demand", datetime


def livestock_meat_dataset():
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
    df = preprocess(df, "AMOUNT")
    return df, "AMOUNT", pd.Series(list(range(len(df))))


def online_retail_dataset():
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail" / "online_retail.xlsx")
    df.dropna(inplace=True)
    # fix dtypes
    df['CustomerID'] = df['CustomerID'].astype('int64').astype('category')
    df['Country'] = df['Country'].astype('category')
    df['StockCode'] = df['StockCode'].astype('category')
    # datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.sort_values(by=['InvoiceDate'], inplace=True)
    invoice_date = df.InvoiceDate.copy()
    # preprocess
    df = preprocess(df, "Quantity", "InvoiceDate")
    # drop unusable columns
    df.drop(columns=['InvoiceNo', 'Description'], inplace=True)
    # TODO Convert 'Description' feature to categories

    return df, "Quantity", invoice_date


def online_retail_2_dataset():
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail_II_dataset" / "online_retail_II.xlsx",
                       sheet_name=['Year 2009-2010', 'Year 2010-2011'])

    df = pd.concat([df['Year 2009-2010'], df['Year 2010-2011']])
    df.dropna(inplace=True)
    # fix dtypes
    df['Customer ID'] = df['Customer ID'].astype('int64').astype('category')
    df['Country'] = df['Country'].astype('category')
    df['StockCode'] = df['StockCode'].astype('category')
    # datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.sort_values(by=['InvoiceDate'], inplace=True)
    invoice_date = df.InvoiceDate.copy()
    # preprocess
    df = preprocess(df, "Quantity", "InvoiceDate")
    # drop unusable columns
    df.drop(columns=['InvoiceNo', 'Description'], inplace=True)
    # TODO Convert 'Description' feature to categories

    return df, "Quantity", invoice_date
