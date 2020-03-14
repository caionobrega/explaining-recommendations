import numpy as np
import pandas as pd
import sklearn
from lime import lime_base, explanation
from sklearn.utils import check_random_state

from src.dataset import Dataset


class LimeRSExplainer():

    def __init__(self,
                 training_df,
                 feature_names,
                 feature_map,
                 mode="classification",
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 random_state=None):

        # exponential kernel
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel, verbose,
                                       random_state=self.random_state)

        self.feature_names = list(feature_names)
        self.feature_map = feature_map
        self.mode = mode
        self.class_names = class_names
        self.feature_selection = feature_selection

        self.categorical_features = list(range(feature_names.shape[0]))

        self.n_rows = training_df.shape[0]
        self.training_df = training_df
        self.user_freq = training_df['user_id'].value_counts(normalize=True)
        self.item_freq = training_df['item_id'].value_counts(normalize=True)

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    def explain_instance(self,
                         instance,
                         rec_model,
                         neighborhood_entity,
                         labels=(1,),
                         num_features=10,
                         num_samples=50,
                         distance_metric='cosine',
                         model_regressor=None):

        # get neighborhood
        neighborhood_df = self.generate_neighborhood(instance, neighborhood_entity, num_samples)

        # compute distance based on interpretable format
        data, _ = Dataset.convert_to_pyfm_format(neighborhood_df, columns=rec_model.one_hot_columns)
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        # get predictions from original complex model
        yss = np.array(rec_model.predict(neighborhood_df))

        # for classification, the model needs to provide a list of tuples - classes along with prediction probabilities
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                            numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        ret_exp = explanation.Explanation(domain_mapper=None,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            raise NotImplementedError("LIME-RS does not currently support classifier models.")
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp

    def generate_neighborhood(self, instance, entity, num_samples):
        samples = list()
        samples.append({"user_id": str(instance.user_id), "item_id": str(instance.item_id)})
        if entity == 'user':
            sample_users = np.random.choice(self.user_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.user_freq.values.tolist())
            for u in sample_users:
                samples.append({"user_id": str(u), "item_id": str(instance.item_id)})

        elif entity == 'item':
            sample_items = np.random.choice(self.item_freq.index.tolist(), num_samples - 1, replace=False,
                                            p=self.item_freq.values.tolist())
            for i in sample_items:
                samples.append({"user_id": str(instance.user_id), "item_id": str(i)})
        else:
            sample_rows = np.random.choice(range(self.n_rows), num_samples - 1, replace=False)
            for s in self.training_df.iloc[sample_rows].itertuples():
                samples.append({"user_id": str(s.user_id), "item_id": str(s.item_id)})

        samples_df = pd.DataFrame(samples)
        samples_df = samples_df[['user_id', 'item_id']]

        return samples_df
