import os

from src import data_utils
from experiment import utils

logger = utils.get_logger("predict")


def get_predictions(rec_model, test_df, predictions_filename):
    # calculate
    predictions = rec_model.predict(test_df)
    test_df['prediction'] = predictions

    # write
    predictions_df = test_df.sort_values(by=['user_id', 'prediction'], ascending=[True, False])
    predictions_df = predictions_df[['user_id', 'item_id', 'prediction']]
    predictions_df.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, predictions_filename),
                          sep='\t', index=False, header=True)


def get_recs(rec_model, selected_users, recs_filename):
    # calculate
    recs = rec_model.recommend(selected_users)

    # write
    recs.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, recs_filename),
                sep='\t', index=False, header=True)


def main():
    # setup
    exp_setup = utils.setup()

    # load data and rec model
    logger.info("Load data and recommender")
    dataset = data_utils.load_data()
    rec_model = data_utils.load_dump(exp_setup.rec_name)

    # calculate
    logger.info("Generate predictions")
    get_predictions(rec_model, dataset.test_df, "predictions-{}".format(exp_setup.rec_name))

    logger.info("Generate recommendations")
    selected_users = ["1"]
    get_recs(rec_model, selected_users, "recs-{}".format(exp_setup.rec_name))


if __name__ == "__main__":
    main()
