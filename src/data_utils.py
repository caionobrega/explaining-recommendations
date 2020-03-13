import os
import pickle
from os.path import expanduser

import numpy as np
import pandas as pd

# path constants
from src.dataset import Dataset

HOME = os.path.join(expanduser("~"), "sac2019")
DEFAULT_DATA_FOLDER = os.path.join(HOME, "data")
DEFAULT_OUTPUT_FOLDER = os.path.join(HOME, "output")

predictions_filename = "predictions"
recs_filename = "recs"


def load_data():
    # read
    training_df = pd.read_csv(os.path.join(DEFAULT_DATA_FOLDER, "training"), sep="\t",
                              dtype={"user_id": str, "item_id": str})
    test_df = pd.read_csv(os.path.join(DEFAULT_DATA_FOLDER, "test"), sep="\t",
                          dtype={"user_id": str, "item_id": str})
    item_info_long = pd.read_csv(os.path.join(DEFAULT_DATA_FOLDER, "item_features"), sep="\t",
                                 dtype={"item_id": str})
    item_info_wide = item_info_long.pivot(index="item_id", columns="feature", values="value").reset_index().fillna(0)

    #
    y_train = training_df.rating.values.astype(np.float)
    training_df = training_df.drop(columns=["rating"])

    y_test = test_df.rating.values.astype(np.float)
    test_df = test_df.drop(columns=["rating"])

    return Dataset(training_df, y_train, test_df, y_test, item_info_wide)


def load_predictions(rec_name):
    input_filename = "{}-{}".format(predictions_filename, rec_name)
    predictions_df = pd.read_csv(os.path.join(DEFAULT_OUTPUT_FOLDER, input_filename), sep="\t",
                                 dtype={"user_id": str, "item_id": str})

    return predictions_df


def load_recs(rec_name):
    input_filename = "{}-{}".format(recs_filename, rec_name)
    recs_df = pd.read_csv(os.path.join(DEFAULT_OUTPUT_FOLDER, input_filename), sep="\t",
                          dtype={"user_id": str, "item_id": str})

    return recs_df


def save_recs(df, rec_name):
    output_filename = "{}-{}".format(recs_filename, rec_name)
    df.to_csv(path_or_buf=os.path.join(DEFAULT_OUTPUT_FOLDER, output_filename), sep='\t', index=False, header=True)


def save_predictions(df, rec_name):
    output_filename = "{}-{}".format(predictions_filename, rec_name)

    predictions_df = df.sort_values(by=['user_id', 'prediction'], ascending=[True, False])
    predictions_df = predictions_df[['user_id', 'item_id', 'prediction']]
    predictions_df.to_csv(path_or_buf=os.path.join(DEFAULT_OUTPUT_FOLDER, output_filename),
                          sep='\t', index=False, header=True)


def write_dump(rec_model, output_filename):
    # create folder
    if not os.path.exists(DEFAULT_OUTPUT_FOLDER):
        os.makedirs(DEFAULT_OUTPUT_FOLDER)

    # save
    with open(os.path.join(DEFAULT_OUTPUT_FOLDER, output_filename), "wb") as output:
        pickle.dump(rec_model, output, pickle.HIGHEST_PROTOCOL)


def load_dump(input_filename):
    with open(os.path.join(DEFAULT_OUTPUT_FOLDER, input_filename), "rb") as input_file:
        rec_model = pickle.load(input_file)

    return rec_model
