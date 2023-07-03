import logging
import os

from utils.config import cfg


def set_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_logger():
    
    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    if cfg.MODEL.STAGE_TYPE == 4:
        logger = set_logger('./logs/{}-train_on_{}-{}-stage-test.log'
                            .format(cfg.MODEL.NAME, cfg.DATA.TRAIN_DATASET, cfg.DATA.TEST_DATASET))
    else:
        logger = set_logger('./logs/{}-train_on_{}-stage-{}.log'
                            .format(cfg.MODEL.NAME, cfg.DATA.TRAIN_DATASET, cfg.MODEL.STAGE_TYPE))
    
    return logger

