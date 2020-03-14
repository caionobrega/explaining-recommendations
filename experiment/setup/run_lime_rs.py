import json
import os

import numpy as np
import pandas as pd

from experiment import utils
from src import data_utils
from src.lime_rs import LimeRSExplainer

logger = utils.get_logger("limers")


def extract_features(explanation_all_ids, feature_type, feature_map):
    filtered_dict = dict()
    if feature_type == "features":
        for tup in explanation_all_ids:
            if not (feature_map[tup[0]].startswith('user_id') or
                    feature_map[tup[0]].startswith('item_id')):
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    elif feature_type == "item":
        top_features = 500
        for tup in explanation_all_ids:
            if feature_map[tup[0]].startswith('item_id') and len(filtered_dict <= top_features):
                filtered_dict[feature_map[tup[0]]] = round(tup[1], 3)

    return filtered_dict


def generate_explanations(instances_to_explain, explainer, rec_model, feature_type='features'):
    result = list()

    for instance in instances_to_explain.itertuples(index=False):
        logger.info("explaining-> (user: {}, item: {})".format(instance.user_id, instance.item_id))

        exp = explainer.explain_instance(instance,
                                         rec_model,
                                         neighborhood_entity="item",
                                         labels=[0],
                                         num_samples=1000)

        # filter
        filtered_features = extract_features(exp.local_exp[0],
                                             feature_type=feature_type,
                                             feature_map=explainer.feature_map)
        #
        explanation_str = json.dumps(filtered_features)
        output_df = pd.DataFrame({'user_id': [instance.user_id], 'item_id': [instance.item_id],
                                  'explanations': [explanation_str],
                                  'local_prediction': [round(exp.local_pred[0], 3)]})

        result.append(output_df)

    return pd.concat(result)


def main():
    # setup
    exp_setup = utils.setup()

    # load data and rec model
    logger.info("Load data and recommender")
    rec_model = data_utils.load_dump(exp_setup.rec_name)

    # setup explainer
    feature_names = rec_model.one_hot_columns
    feature_map = {i: rec_model.one_hot_columns[i] for i in range(len(list(rec_model.one_hot_columns)))}
    explainer = LimeRSExplainer(rec_model.dataset.training_df,
                                feature_names=feature_names,
                                feature_map=feature_map,
                                mode='regression',
                                class_names=np.array(['rec']),
                                feature_selection='none')

    #
    instances_to_explain = pd.DataFrame([("1", "5")], columns=["user_id", "item_id"])
    explanations = generate_explanations(instances_to_explain, explainer, rec_model)

    # save
    logger.info("Save LimeRS explanations")
    output_filename = "limers_explanations-{}".format(exp_setup.rec_name)
    explanations.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename),
                        sep='\t', index=False, header=True)


if __name__ == '__main__':
    main()
