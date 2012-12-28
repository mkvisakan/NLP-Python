import os, sys
from languageModels import compute_cross_entropy_perplexity, Unigram, Bigram, Trigram, InterpolatedModel, sentenceGenerator
from orwellDataParser import OrwellDataParser
from loggerUtils import initialize_logger
import config

def main(logger):
    config.log_inputs(logger)
    data_parser = OrwellDataParser(logger, config.STOP_TOKEN)
    print "===================================================="
    print "UNIGRAM MODEL"
    print "===================================================="
    unigram_obj = Unigram(logger, data_parser, config.VOCABULARY_COUNT, config.STOP_TOKEN, config.S_FACTOR)
    unigram_obj.train(config.TRAIN_DATA)
    u_tr_cross_entropy, u_tr_perplexity = compute_cross_entropy_perplexity(unigram_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % u_tr_cross_entropy
    print "\t Perplexity    : %.10f" % u_tr_perplexity
    u_ts_cross_entropy, u_ts_perplexity = compute_cross_entropy_perplexity(unigram_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % u_ts_cross_entropy
    print "\t Perplexity    : %.10f" % u_ts_perplexity
    print "===================================================="

    print "===================================================="
    print "BIGRAM MODEL"
    print "===================================================="
    bigram_obj = Bigram(logger, data_parser, config.VOCABULARY_COUNT, config.STOP_TOKEN, config.S_FACTOR)
    bigram_obj.train(config.TRAIN_DATA)
    b_tr_cross_entropy, b_tr_perplexity = compute_cross_entropy_perplexity(bigram_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % b_tr_cross_entropy
    print "\t Perplexity    : %.10f" % b_tr_perplexity
    b_ts_cross_entropy, b_ts_perplexity = compute_cross_entropy_perplexity(bigram_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % b_ts_cross_entropy
    print "\t Perplexity    : %.10f" % b_ts_perplexity
    print "===================================================="


    print "===================================================="
    print "TRIGRAM MODEL"
    print "===================================================="
    trigram_obj = Trigram(logger, data_parser, config.VOCABULARY_COUNT, config.STOP_TOKEN, config.S_FACTOR)
    trigram_obj.train(config.TRAIN_DATA)
    t_tr_cross_entropy, t_tr_perplexity = compute_cross_entropy_perplexity(trigram_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % t_tr_cross_entropy
    print "\t Perplexity    : %.10f" % t_tr_perplexity
    t_ts_cross_entropy, t_ts_perplexity = compute_cross_entropy_perplexity(trigram_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % t_ts_cross_entropy
    print "\t Perplexity    : %.10f" % t_ts_perplexity
    print "===================================================="

    print "===================================================="
    print "INTERPOLATED MODEL"
    print "===================================================="
    i_obj = InterpolatedModel(logger, data_parser, unigram_obj, bigram_obj, trigram_obj)
    i_tr_cross_entropy, i_tr_perplexity = compute_cross_entropy_perplexity(i_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % i_tr_cross_entropy
    print "\t Perplexity    : %.10f" % i_tr_perplexity
    i_ts_cross_entropy, i_ts_perplexity = compute_cross_entropy_perplexity(i_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % i_ts_cross_entropy
    print "\t Perplexity    : %.10f" % i_ts_perplexity
    print "===================================================="

    print "===================================================="
    print "NEW INTERPOLATED MODEL"
    print "===================================================="
    n_i_obj = InterpolatedModel(logger, data_parser, unigram_obj, bigram_obj, trigram_obj, 32.0/100, 33.0/100, 35.0/100.0)
    n_i_tr_cross_entropy, n_i_tr_perplexity = compute_cross_entropy_perplexity(n_i_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % n_i_tr_cross_entropy
    print "\t Perplexity    : %.10f" % n_i_tr_perplexity
    n_i_ts_cross_entropy, n_i_ts_perplexity = compute_cross_entropy_perplexity(n_i_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % n_i_ts_cross_entropy
    print "\t Perplexity    : %.10f" % n_i_ts_perplexity
    print "===================================================="

    '''
    print "===================================================="
    print "INTERPOLATED MODEL"
    print "===================================================="
    i_obj = InterpolatedModel(logger, data_parser, unigram_obj, bigram_obj, trigram_obj, (1.0/10.0), (4.4/10.0), (4.6/10.0))
    i_tr_cross_entropy, i_tr_perplexity = compute_cross_entropy_perplexity(i_obj, config.TRAIN_DATA)
    print "TRAIN_DATA     : "
    print "\t Cross Entropy : %.10f" % i_tr_cross_entropy
    print "\t Perplexity    : %.10f" % i_tr_perplexity
    i_ts_cross_entropy, i_ts_perplexity = compute_cross_entropy_perplexity(i_obj, config.TEST_DATA)
    print "TEST_DATA     : "
    print "\t Cross Entropy : %.10f" % i_ts_cross_entropy
    print "\t Perplexity    : %.10f" % i_ts_perplexity
    print "===================================================="
    '''

    print "===================================================="
    print "UNIGRAM : sentence generation"
    print "===================================================="
    u_s_obj = sentenceGenerator(logger, unigram_obj, config.STOP_TOKEN)
    print u_s_obj.generate()

    
    print "===================================================="
    print "BIGRAM : sentence generation"
    print "===================================================="
    b_s_obj = sentenceGenerator(logger, bigram_obj, config.STOP_TOKEN)
    print b_s_obj.generate()

    print "===================================================="
    print "TRIGRAM : sentence generation"
    print "===================================================="
    t_s_obj = sentenceGenerator(logger, trigram_obj, config.STOP_TOKEN)
    print t_s_obj.generate()

    print "===================================================="
    print "INTERPOLATED : sentence generation"
    print "===================================================="
    i_s_obj = sentenceGenerator(logger, i_obj, config.STOP_TOKEN)
    print i_s_obj.generate()
    
if __name__ == "__main__":
    logger  = initialize_logger(os.path.splitext(sys.argv[0])[0]+ os.path.extsep + "log")    
    main(logger)
