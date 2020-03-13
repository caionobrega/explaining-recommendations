from experiment import utils
from src import data_utils
from src.model import FMRec

logger = utils.get_logger("train_rec")


def main():
    # setup
    exp_setup = utils.setup()

    # load
    logger.info("Load data")
    dataset = data_utils.load_data()

    # train
    logger.info("Train FM model")
    rec_model = FMRec(rec_name=exp_setup.rec_name, dataset=dataset, uses_features=exp_setup.uses_features)
    rec_model.train()

    # write dump
    logger.info("Save rec model dump")
    data_utils.write_dump(rec_model, exp_setup.rec_name)


if __name__ == "__main__":
    main()
