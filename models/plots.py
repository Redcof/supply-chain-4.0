import logging

import pandas as pd
from matplotlib import pyplot as plt, rcParams

logger = logging.getLogger(__file__)


def plots(prefix, y_test, y_pred, x):
    rcParams['font.family'] = 'serif'
    df = pd.DataFrame(dict(y=y_test, y_pred=y_pred, timeline=x))
    ax = df.plot.line(x='timeline', figsize=(20, 10))
    fig = ax.get_figure()
    file_name = "%s.png" % prefix
    fig.savefig(file_name)
    # fig.savefig("%s_%s.png" % (prefix, datetime.now()))
    logger.info(f"Plot is saved at {file_name}")
    plt.close()
