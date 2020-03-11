from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pyfm import pylibfm
from scipy import sparse


class Model(ABC):
    def __init__(self, uses_features):
        self.uses_features = uses_features

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, df):
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id, training_df, n=10, filter_history=True):
        raise NotImplementedError

    def __get_candidates(self, user_id, training_df, filter_history=True):
        all_items = set(training_df["item_id"].values.tolist())
        user_items = set(training_df.loc[training_df["user_id"] == str(user_id)]["item_id"].values.tolist())
        candidate_items = all_items
        if filter_history:
            candidate_items = all_items - user_items

        # create dataframe
        df = pd.DataFrame()
        df['item_id'] = list(candidate_items)
        df['user_id'] = str(user_id)
        df = df[['user_id', 'item_id']]

        return df


class FMRec(Model):
    def __init__(self, uses_features):
        super(FMRec, self).__init__(uses_features)
        self.fm = pylibfm.FM(num_factors=50, num_iter=10, task="regression",
                             initial_learning_rate=0.001,
                             learning_rate_schedule="optimal",
                             verbose=True)
        self.training_columns = None

    def train(self, training_df, y_train):
        training_data, training_columns = self.__convert_to_pyfm_format(training_df,
                                                                        uses_features=self.uses_features)
        self.fm.fit(training_data, y_train)
        self.training_columns = training_columns

    def predict(self, df):
        all_predictions = list()

        # divide in chunks to avoid memory errors
        chunks = np.array_split(df, 10)
        for chunck in chunks:
            # convert
            test_data, _ = self.__convert_to_pyfm_format(chunck,
                                                         uses_features=self.uses_features,
                                                         columns=self.training_columns)
            # get predictions
            preds = self.fm.predict(test_data)
            all_predictions.extend(preds.round(3))

        return all_predictions

    def recommend(self, user_ids, training_df, n=10, filter_history=True):
        df_list = list()

        # get predictions
        for uid in user_ids:
            print(uid)
            df = self._get_candidates(uid, training_df, filter_history=filter_history)
            predictions = self.predict(df)
            df['prediction'] = predictions

            df_list.append(df)
        result = pd.concat(df_list)

        # Sort by uid and predictions
        result.sort_values(by=['user_id', 'prediction'], inplace=True, ascending=[True, False])
        return result.groupby("user_id").head(n)

    # noinspection PyMethodMayBeStatic
    def __convert_to_pyfm_format(self, df, uses_features=True, columns=None):
        if uses_features:
            df_sparse = df.to_sparse()
            df_ohe = pd.get_dummies(df_sparse, sparse=True)
            if columns is not None:
                df_ohe = df_ohe.reindex(columns=columns)
            data_sparse = sparse.csr_matrix(df_ohe.astype(np.float).to_coo())
            data_sparse = data_sparse.astype(np.float)
        else:
            df_ohe = pd.get_dummies(df, sparse=False)
            if columns is not None:
                df_ohe = df_ohe.reindex(columns=columns)
            df_ohe = df_ohe.fillna(0)
            data_sparse = sparse.csr_matrix(df_ohe.astype(np.float))

        data_sparse = data_sparse.astype(np.float)

        # check
        # nonzero_elements = df_ohe.columns.columns_ids[data_sparse.iloc[1].nonzero()[0]].values.tolist()

        return data_sparse, df_ohe.columns
