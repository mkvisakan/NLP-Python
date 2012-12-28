VOCABULARY_COUNT    = 9197
STOP_TOKEN          = "<S>"
S_FACTOR            = 0.00001
TEST_DATA           = "hw3-data/orwell-test.txt"
TRAIN_DATA          = "hw3-data/orwell-train.txt"


def log_inputs(logger):
    logger.info("****************************************");
    logger.info("VOCABULARY_COUNT        : %d" % VOCABULARY_COUNT);
    logger.info("STOP_TOKEN              : %s" % STOP_TOKEN);
    logger.info("S_FACTOR                : %.5f" % S_FACTOR);
    logger.info("TRAIN_DATA              : %s" % TRAIN_DATA);
    logger.info("TEST_DATA               : %s" % TEST_DATA);
    logger.info("****************************************");
    
