import logging
from collections import namedtuple

# experiment config
ExperimentSetup = namedtuple('Rec', ['rec_name', 'uses_features'])
exp_setups = dict()
SETUP = 'setup1'


def get_logger(name=None):
    # setup logger
    logger_name = '' if name is None else str(name)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sac2019:{}".format(logger_name))
    logger.setLevel(logging.INFO)

    return logger


def setup():
    def config_setups():
        if len(exp_setups) == 0:
            rec_filename = 'fm_regressor-k_50'
            exp_setups['setup1'] = ExperimentSetup(rec_filename, False)
            exp_setups['setup2'] = ExperimentSetup('{}-movie_genres'.format(rec_filename), True)

    config_setups()
    return exp_setups[SETUP]
