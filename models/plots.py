import pandas as pd
from matplotlib import pyplot as plt


def plots(y_test, y_pred):
    df = pd.DataFrame(dict(y=y_test, y_pred=y_pred))
    df.plot.line()
    plt.show()
