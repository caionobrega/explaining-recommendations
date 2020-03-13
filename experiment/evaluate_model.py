import pandas as pd

from experiment import utils
from src import data_utils, evaluation

logger = utils.get_logger("compute_metrics")


def compute_accuracy_metrics(group, test_df):
    # get data
    user_id = group['user_id'].unique()[0]
    rec_list = group['item_id'].tolist()
    test_list = test_df.loc[test_df['user_id'] == user_id]['item_id'].tolist()

    if len(test_list) == 0:
        raise ValueError("user_id: {} does not exist in the test set".format(user_id))

    #
    precision_value = evaluation.compute_precision(rec_list, test_list)
    recall_value = evaluation.compute_recall(rec_list, test_list)

    return user_id, precision_value, recall_value


def main():
    # setup
    exp_setup = utils.setup()

    # load
    logger.info("Load data")
    dataset = data_utils.load_data()
    predictions = data_utils.load_predictions(exp_setup.rec_name)
    recs = data_utils.load_recs(exp_setup.rec_name)

    # calculate
    logger.info("Compute metrics")
    dataset.test_df['rating'] = dataset.y_test
    merged = pd.merge(dataset.test_df, predictions, on=["user_id", "item_id"], how="left")
    test_rmse_results = evaluation.compute_rmse(merged['rating'], merged['prediction'])
    logger.info("rec: {} - Test RMSE results: {}".format(exp_setup.rec_name, round(test_rmse_results, 3)))

    # accuracy
    acc_results = recs.groupby('user_id').apply(compute_accuracy_metrics, dataset.test_df).values.tolist()
    acc_results_df = pd.DataFrame(acc_results, columns=['user_id', 'precision', 'recall'])
    logger.info("{} - Accuracy results:\n{}".format(exp_setup.rec_name, acc_results_df))
    logger.info("{} - Average Precision: {}".format(exp_setup.rec_name, round(acc_results_df['precision'].mean(), 3)))


if __name__ == "__main__":
    main()
