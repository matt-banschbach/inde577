import pandas as pd
import sklearn.datasets as skd

def load_wine():
    """
    Wrapper for sklearn.datasets.load_wine; processes the data to a DataFrame object, as opposed to a 
    :return:
    """
    wine_data = skd.load_wine()
    wine = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    wine['target'] = wine_data.target
    return wine