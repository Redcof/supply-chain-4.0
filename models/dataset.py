import pathlib

import pandas as pd

root_dir = pathlib.Path(r"/Users/soumensardar/Library/CloudStorage/GoogleDrive-soumensardarintmain@gmail.com/"
                        r".shortcut-targets-by-id/1zlA8VX-l_IufCZSfJIfxohpsEGrcLx4-/Taniya Paul Thesis paper/"
                        r"writing/Research Proposal/dataset")


def forecasts_for_product_demand_dataset():
    df = pd.read_csv(root_dir / "Forecasts for Product Demand-base.csv")

    # treat Order_Demand
    df['Order_Demand'] = df['Order_Demand'].str.replace('(', "")
    df['Order_Demand'] = df['Order_Demand'].str.replace(')', "")
    df['Order_Demand'] = df['Order_Demand'].astype('int64')
    # fix dtypes
    df['Product_Code'] = df['Product_Code'].astype('category')
    df['Warehouse'] = df['Warehouse'].astype('category')
    df['Product_Category'] = df['Product_Category'].astype('category')
    df.drop(columns=['Date'], inplace=True)

    return df, "Order_Demand"
