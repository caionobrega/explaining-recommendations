def compute_precision(rec_list, test_list):
    intersection = set(rec_list) & set(test_list)
    return round(len(intersection) / len(rec_list), 3)


def compute_recall(rec_list, test_list):
    intersection = set(rec_list) & set(test_list)
    return round(len(intersection) / len(test_list), 3)


def compute_rmse(ratings, predictions):
    rmse = ((ratings - predictions) ** 2).mean() ** .5
    return rmse
