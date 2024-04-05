from models import xgboost_wrap
from models.dataset import forecasts_for_product_demand_dataset


def main():
    df, target = forecasts_for_product_demand_dataset()
    xgboost_wrap.trainer(df, target=target)


if __name__ == '__main__':
    main()
