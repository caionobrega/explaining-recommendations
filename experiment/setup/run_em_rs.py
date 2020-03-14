import os

import pandas as pd

from experiment import utils
from src import data_utils, em_rs

logger = utils.get_logger("emrs")


def get_recs_from_rules(training, rules, users_to_recommend, top_n=10):
    result = list()
    for user in users_to_recommend:
        logger.info(user)
        user_items = set(training[training['userId'] == user]['movieId'].values.tolist())

        part1 = rules[(rules['item_A'].isin(user_items)) & (~rules['item_B'].isin(user_items))]
        part1 = part1[['item_A', 'item_B', 'support', 'confidence', 'lift']]
        part1['userId'] = user
        part1.columns = ['training_item', 'rec_item', 'support', 'confidence', 'lift', 'userId']

        # get top
        recs_by_confidence = part1.sort_values(by=['userId', 'confidence'], ascending=[True, False])
        recs_by_confidence = recs_by_confidence.drop_duplicates(subset=['userId', 'rec_item'], keep='first')[
            ['userId', 'rec_item', 'confidence', 'training_item']]
        recs_by_confidence = recs_by_confidence.head(top_n)
        result.append(recs_by_confidence)

    return pd.concat(result)


def main():
    # setup
    exp_setup = utils.setup()

    # load data and rec model
    logger.info("Load data and recommender")
    dataset = data_utils.load_data()

    # experiment global: AR - top30 FM score as input transaction
    recs = data_utils.load_recs("{}-top30".format(exp_setup.rec_name))
    recs = recs.groupby('user_id')['item_id'].apply(list).values.tolist()

    # train ar
    logger.info("Get EM-RS explanations")
    apriori_config = {"min_support": 0.1, "min_confidence": 0.1, "min_lift": 0.1, "max_length": 2}
    rules = em_rs.calculate_apriori(recs, apriori_config)

    # get recs
    users_to_recommend = ['1']
    top_n = 10
    em_recs = get_recs_from_rules(dataset.training_df, rules, users_to_recommend, top_n)

    # write
    logger.info("Save EM-RS explanations")
    output_filename = "emrs_explanations"
    em_recs.to_csv(path_or_buf=os.path.join(data_utils.DEFAULT_OUTPUT_FOLDER, output_filename),
                   sep='\t', index=False, header=True)


if __name__ == '__main__':
    main()
