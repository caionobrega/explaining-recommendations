from experiment import utils
from src import data_utils

logger = utils.get_logger("predict")


def main():
    # setup
    exp_setup = utils.setup()

    # load data and rec model
    logger.info("Load data and recommender")
    dataset = data_utils.load_data()
    rec_model = data_utils.load_dump(exp_setup.rec_name)

    # calculate predictions
    logger.info("Generate predictions")
    predictions = rec_model.predict(dataset.test_df)
    dataset.test_df['prediction'] = predictions
    data_utils.save_predictions(dataset.test_df, exp_setup.rec_name)

    # calculate recs
    logger.info("Generate recommendations")
    selected_users = ["1"]
    recs = rec_model.recommend(selected_users)
    data_utils.save_recs(recs, exp_setup.rec_name)


if __name__ == "__main__":
    main()
