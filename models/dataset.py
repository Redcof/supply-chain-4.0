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
    df['center_id'] = df['center_id'].astype('category')
    df['meal_id'] = df['meal_id'].astype('category')
    df['emailer_for_promotion'] = df['emailer_for_promotion'].astype('category')
    df['homepage_featured'] = df['homepage_featured'].astype('category')

    df.sort_values(by=['week'], inplace=True)
    df = preprocess(df, "num_orders")
    return df, "num_orders", df.index


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
    df.sort_values(by=['Date'], inplace=True)
    df.drop(columns=['Date'], inplace=True)

    return df, "Order_Demand"


def livestock_meat_dataset():
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_csv(root_dir / "livestock_meat" / "livestock_meat_imports.csv")
    df.dropna(inplace=True)
    df.drop(columns=["COMMODITY_DESC", "SOURCE_ID", "ATTRIBUTE_DESC"], inplace=True)
    df.sort_values(by=['YEAR_ID', 'TIMEPERIOD_ID'], inplace=True)

    return df, "AMOUNT"


def online_retail_dataset():
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail" / "online_retail.xlsx")
    df.dropna(inplace=True)
    # treat Order_Demand
    df['Order_Demand'] = df['Order_Demand'].str.replace('(', "")
    df['Order_Demand'] = df['Order_Demand'].str.replace(')', "")
    df['Order_Demand'] = df['Order_Demand'].astype('int64')
    # fix dtypes
    df['CustomerID'] = df['CustomerID'].astype('int64').astype('category')
    df['Country'] = df['Country'].astype('category')
    df['StockCode'] = df['StockCode'].astype('category')

    # drop unusable columns
    df.drop(columns=['InvoiceNo', 'Description'], inplace=True)
    # TODO Convert Description feature to categories

    return df, "Quantity", "InvoiceDate"


def online_retail_II_dataset():
    """
    Returns a tuple (cleaned dataset in pandas dataframe, the target column name)
    :return:
    """
    df = pd.read_excel(root_dir / "online_retail_II_dataset" / "online_retail_II.xlsx",
                       sheet_name=['Year 2009-2010', 'Year 2010-2011'])

    df = pd.concat([df['Year 2009-2010'], df['Year 2010-2011']])
    # treat Order_Demand
    df.dropna(inplace=True)
    # fix dtypes
    # df['Product_Code'] = df['Product_Code'].astype('category')
    # df['Warehouse'] = df['Warehouse'].astype('category')
    # df['Product_Category'] = df['Product_Category'].astype('category')
    df.drop(columns=['Date'], inplace=True)

    return df, "Order_Demand"
