import os, sys
from loggerUtils import initialize_logger
from orwellDataParser import OrwellDataParser
from linear_sequence import LinearSequence
from viterbi import Viterbi
from accuracyEstimator import AccuracyEstimator
import datetime

ENGLISH_TRAINING_FILE   = "Data/orwell-en.txt_formatted"
TRAINING_FILES          = [("ENGLISH" , "orwell-en.txt_formatted") , ("CZECH" , "orwell-cs.txt_formatted"), ("FARSI" , "orwell-fa.txt_formatted"), ("HUNGARIAN" , "orwell-hu.txt_formatted"), ("SLOVENE" , "orwell-sl.txt_formatted"),\
                           ("BULGARIAN" , "orwell-bg.txt_formatted"), ("ESTONIAN", "orwell-et.txt_formatted"), ("POLISH", "orwell-pl.txt_formatted"), ("ROMANIAN", "orwell-ro.txt_formatted"), ("SLOVAKIAN", "orwell-sk.txt_formatted"),\
                           ("SERBIAN", "orwell-sr.txt_formatted")
                          ]
START_LINE              = 5501

DATA_FILES              = [ ("ENGLISH", ENGLISH_TRAINING_FILE) ]

OUTPUT                  = "STATS.txt"

def main(logger):

    out_ptr             = open(OUTPUT, "w")
    print "Initializing Data Parser..."
    data_parser         = OrwellDataParser(logger)

    print "Initializing Linear Sequence Model..."
    ls_obj              = LinearSequence(logger, data_parser)

    print "Initializing Viterbi..."
    viterbi_obj         = Viterbi(logger, ls_obj)

    print "Initializing Accuracy Estimator..."
    accuracy_estimator  = AccuracyEstimator(logger, data_parser)

    for language, language_file in TRAINING_FILES:
        language_file   = "Data/%s" % language_file
        print "******************************************************************************************"
        print language
        print "******************************************************************************************"

        print "Training Linear Sequence Linear Sequence Model with %s data..." % language
        viterbi_obj.train(language_file, START_LINE - 1)

        #import pdb;pdb.set_trace()
        #print viterbi_obj.predict_sequence(["his", "breast", "rose", "and", "fell", "a", "little", "faster", "."])

        print "Estimating accuracy of the model..."
        total_accuracy, unseen_accuracy = accuracy_estimator.compute_parameters(viterbi_obj, language_file, "inaccurate_words_%s.txt"%language, language, START_LINE)

        print "TOTAL ACCURACY           : %.10f" % total_accuracy
        print "UNSEEN_ACCURACY          : %.10f" % unseen_accuracy

        o_line = "Language : %s\nTotal Accuracy : %.10f\nUnseen Accuracy : %.10f\n\n\n" % (language, total_accuracy, unseen_accuracy)
        out_ptr.write(o_line)
        print "Resetting model and estimator parameters..."

        viterbi_obj.reset()
        accuracy_estimator.reset()

    out_ptr.close()


if __name__ == "__main__":
    today  = datetime.datetime.now()
    logger = initialize_logger(os.path.splitext(sys.argv[0])[0]+ today.strftime("%Y%m%d_%H%M%S") + os.path.extsep + "log")
    main(logger)
