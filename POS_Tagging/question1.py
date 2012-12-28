import os, sys
from loggerUtils import initialize_logger
from orwellDataParser import OrwellDataParser
from simpleTagger import SimpleTagger
from accuracyEstimator import AccuracyEstimator

ENGLISH_TRAINING_FILE   = "hw4-data/orwell-en.txt"
CZECH_TRAINING_FILE     = "hw4-data/orwell-cs.txt"
POLISH_TRAINING_FILE    = "hw4-data/orwell-pl.txt"
START_LINE              = 5501

DATA_FILES              = [ ("ENGLISH", ENGLISH_TRAINING_FILE), ("CZECH", CZECH_TRAINING_FILE), ("POLISH", POLISH_TRAINING_FILE) ]


def main(logger):
    
    print "Initializing Data Parser..."
    data_parser         = OrwellDataParser(logger)
    
    print "Initializing Simple Tagger Model..."
    simple_tagger       = SimpleTagger(logger, data_parser)
    
    print "Initializing Accuracy Estimator..."
    accuracy_estimator  = AccuracyEstimator(logger, data_parser)

    for language, language_file in DATA_FILES:
        print "******************************************************************************************"
        print language
        print "******************************************************************************************"
        print "Training Simple Tagger with %s data..." % language
        simple_tagger.train(language_file, START_LINE - 1)
        print "Estimating accuracy of the model..."
        total_accuracy, unseen_accuracy = accuracy_estimator.compute_parameters(simple_tagger, language_file, START_LINE)
        print "TOTAL ACCURACY           : %.10f" % total_accuracy
        print "UNSEEN_ACCURACY          : %.10f" % unseen_accuracy
        print "Resetting model and estimator parameters..."
        simple_tagger.reset()
        accuracy_estimator.reset()


if __name__ == "__main__":
    logger = initialize_logger(os.path.splitext(sys.argv[0])[0]+ os.path.extsep + "log")
    main(logger)
