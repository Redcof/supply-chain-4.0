import logging

import mlflow
import pandas as pd
from matplotlib import pyplot as plt, rcParams

logger = logging.getLogger(__name__)


def plots(name, y_test, y_pred, x):
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = '20'
    df = pd.DataFrame(dict(y=y_test, y_pred=y_pred, timeline=x))
    ax = df.plot.line(x='timeline', figsize=(15, 15))
    fig = ax.get_figure()
    file_name = "./output/plots/%s.png" % name
    fig.savefig(file_name)
    logger.info(f"Plot is saved at {file_name}")
    plt.close()
    mlflow.log_artifact(str(file_name), "output/plots")


def save(name, y_test, y_pred, x):
    df = pd.DataFrame(dict(y=y_test, y_pred=y_pred, timeline=x))
    filename = f"./output/csvs/{name}.csv"
    df.to_csv(filename)
    mlflow.log_artifact(filename, "output/csvs")
