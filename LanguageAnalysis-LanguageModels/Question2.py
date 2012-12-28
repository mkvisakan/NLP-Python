import os, sys
from languageModels import compute_cross_entropy_perplexity, Unigram, Bigram, Trigram, InterpolatedModel, sentenceGenerator,\
     GeometricModel, MultinomialModel, NegativeBinomialModel
from orwellDataParser import OrwellDataParser
from loggerUtils import initialize_logger
import config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main(logger):
    config.log_inputs(logger)
    data_parser = OrwellDataParser(logger, config.STOP_TOKEN)


    unigram_obj = Unigram(logger, data_parser, config.VOCABULARY_COUNT, config.STOP_TOKEN, config.S_FACTOR)
    unigram_obj.train(config.TRAIN_DATA)

    q = unigram_obj.word_probability(config.STOP_TOKEN)
    p = 1.0 - q
    
    
    print "===================================================="
    print "GEOMETRIC MODEL"
    print "===================================================="

    g_obj = GeometricModel(logger, data_parser, p, q)

    print "plotting the probability of sentences with length 1 through 100"
    x_list, y_list = g_obj.compute_prob_plots()
    plt.figure("Probability of sentence lengths 1 through 100")
    plt.plot(x_list, y_list, 'b')
    plt.xlabel("sentence_lengths")
    plt.ylabel("probability")
    plt.show()


    print "plotting the frequency of sentence lengths in %s" % config.TRAIN_DATA
    x_list, y_list = g_obj.compute_freq_plots(config.TRAIN_DATA)
    plt.figure("Geometric Model : frequency of sentence lengths in %s" % config.TRAIN_DATA)
    plt.plot(x_list, y_list, 'b')
    plt.xlabel("sentence_lengths")
    plt.ylabel("frequency")
    plt.show()


    print "Cross Entropy and Perplexity of sentence lengths in %s" % config.TEST_DATA
    c, p = g_obj.compute_parameters(config.TEST_DATA)
    print "CROSS ENTROPY    : %.10f" % c
    print "PERPLEXITY       : %.10f" % p

    print "===================================================="
    print "MULTINOMIAL MODEL"
    print "===================================================="
    m_obj = MultinomialModel(logger, data_parser)
    m_obj.train(config.TRAIN_DATA)

    print "probability of sentence lengths 1 through 100"
    x_list, y_list = m_obj.compute_prob_plots()
    plt.figure("Multinomial Model : Probability of sentence lengths 1 through 100")
    plt.plot(x_list, y_list, 'b')
    plt.xlabel("sentence_lengths")
    plt.ylabel("probability")
    plt.show()

    print "Cross Entropy and Perplexity of sentence lengths in %s" % config.TEST_DATA
    c, p = m_obj.compute_parameters(config.TEST_DATA)
    print "CROSS ENTROPY    : %.10f" % c
    print "PERPLEXITY       : %.10f" % p

    print "===================================================="
    print "NEGATIVE BINOMIAL MODEL"
    print "===================================================="
    n_obj = NegativeBinomialModel(logger, data_parser, 0.8682, 2.6878)
    
    print "probability of sentence lengths 1 through 100"
    x_list, y_list = n_obj.compute_prob_plots()
    plt.figure("Negative Binomial Model : Probability of sentence lengths 1 through 100")
    plt.plot(x_list, y_list, 'b')
    plt.xlabel("sentence_lengths")
    plt.ylabel("probability")
    plt.show()

    print "Cross Entropy and Perplexity of sentence lengths in %s" % config.TEST_DATA
    c, p = n_obj.compute_parameters(config.TEST_DATA)
    print "CROSS ENTROPY    : %.10f" % c
    print "PERPLEXITY       : %.10f" % p

    
    



if __name__ == "__main__":
    logger  = initialize_logger(os.path.splitext(sys.argv[0])[0]+ os.path.extsep + "log")    
    main(logger)
