import numpy as np
import pandas as pd
from scipy import sparse


class Dataset:
    def __init__(self, training_df, y_train, test_df, y_test, item_features):
        self.training_df = training_df
        self.y_train = y_train
        self.test_df = test_df
        self.y_test = y_test
        self.item_features = item_features

    @staticmethod
    def convert_to_pyfm_format(df, columns=None):
        df_ohe = pd.get_dummies(df)
        if columns is not None:
            df_ohe = df_ohe.reindex(columns=columns)
            df_ohe = df_ohe.fillna(0)
        data_sparse = sparse.csr_matrix(df_ohe.astype(np.float))
        data_sparse = data_sparse.astype(np.float)

        # check
        # nonzero_elements = df_ohe.columns[data_sparse[0].indices].values.tolist()

        return data_sparse, df_ohe.columns
