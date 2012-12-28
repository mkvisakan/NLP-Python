#
#Author       : Kumaresh Visakan Murugan
#Email        : mvisakan@cs.wisc.edu
#Class        : CS545 - NLP
#Instructor   : Prof. Benjamin Snyder
#

import os, sys
from OrwellDataParser import DataParser, Schema
from loggerUtils import initialize_logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math

class LanguageMeta:
    def __init__(self, logger, language_name, language_in_file):
        self.logger                          = logger
        self.language_name                   = language_name
        self.language_in_file                = language_in_file
        self.sorted_data_list                = []
        self.u_word_types                    = 0
        self.u_lemmas                        = 0
        self.max_lemma                       = None
        self.max_lemma_word_types            = 0
        self.unique_word_type_to_lemma_ratio = 0.0
        self.rank_list                       = []
        self.freq_list                       = []
        self.log_rank_list                   = []
        self.log_freq_list                   = []
        self.log_lemma_rank_list             = []
        self.log_lemma_freq_list             = []
        self.slope                           = 0.0
        self.intercept                       = 0.0


    def analyse_data(self, data_parser, lang_schema):
        self.logger.info("Started analysing %s language data from %s" % (self.language_name, self.language_in_file))
        #iterate and store neccessary data
        lemma_word_dict   = {}
        schema_len        = lang_schema.get_schema_len()
        word_types_dict   = {}
        lemma_freq_dict   = {}
        for data_list in data_parser.iterate(self.language_in_file):

            #validate the data
            if len(data_list) != schema_len:
                self.logger.info("File %s\tInvalid Format : %s" % (language_in_file, "\t".join(data_list)))
                return False

            #extract required data
            word_type     = lang_schema.get_word_type(data_list)
            lemma         = lang_schema.get_lemma(data_list)

            #compute the lemma -> word type count
            word_type_set = lemma_word_dict.setdefault(lemma, set())
            if word_type not in word_type_set:
                word_type_set.add(word_type)

            #compute the lemma -> word type count
            word_freq                  = word_types_dict.setdefault(word_type, 0)
            word_types_dict[word_type] = word_freq + 1

            #compute lemma frequency
            lemma_freq                 = lemma_freq_dict.setdefault(lemma, 0)
            lemma_freq_dict[lemma]     = lemma_freq + 1

        #if Data not found
        if not lemma_word_dict:
            self.logger.info("Data Not found !!!")
            return False

        #sort the lemma -> word type count dict and obtain count of unique word types and unique_lemmas
        self.sorted_data_list                = sorted(lemma_word_dict.iteritems(), key=lambda x:len(x[1]), reverse=True)
        self.u_word_types                    = len(word_types_dict)
        self.u_lemmas                        = len(self.sorted_data_list)
        self.max_lemma                       = self.sorted_data_list[0][0]
        self.max_lemma_word_types            = len(self.sorted_data_list[0][1])
        self.unique_word_type_to_lemma_ratio = float(self.u_word_types)/self.u_lemmas

        #construct plots for this language
        sorted_freq_list = sorted(word_types_dict.iteritems(), key=lambda x:x[1], reverse=True)

        #construct rank_list and freq_list
        for index, (word_type, freq) in enumerate(sorted_freq_list):
            self.log_rank_list.append(math.log(index + 1, 2))
            self.log_freq_list.append(math.log(freq, 2))
            self.rank_list.append(index + 1)
            self.freq_list.append(freq)

        #compute slope, intercept
        self.slope, self.intercept = np.polyfit(self.log_rank_list, self.log_freq_list, 1)

        #compute log lemma rank and log lemma freq data
        sorted_lemma_list = sorted(lemma_freq_dict.iteritems(), key=lambda x:x[1], reverse=True)
        for index, (lemma, lemma_freq) in enumerate(sorted_lemma_list):
            self.log_lemma_rank_list.append(math.log(index + 1, 2))
            self.log_lemma_freq_list.append(math.log(lemma_freq, 2))

        return True


class LanguageAnalyzer:
    def __init__(self, logger, language_info_dict):
        self.logger              = logger
        self.data_parser         = DataParser(self.logger)
        self.lang_schema         = Schema()
        self.language_info_dict  = language_info_dict
        self.lang_out_dir        = "LanguageAnalysisOutput"
        if not os.path.exists(self.lang_out_dir):
            os.makedirs(self.lang_out_dir)
        self.analysed_lang_data  = {}
        self.analyse_languages()

    def analyse_languages(self):
        for language_name, language_in_file in self.language_info_dict.iteritems():
            print "Analyzing %s data files at %s" % (language_name, language_in_file)
            #create language meta object
            language_obj = LanguageMeta(self.logger, language_name, language_in_file)
            status       = language_obj.analyse_data(self.data_parser, self.lang_schema)
            if not status:
                self.terminate_with_error()
            self.analysed_lang_data[language_name] = language_obj


    def analyze_morphology(self):
        morp_out_file_name = os.path.join(self.lang_out_dir, "MorphologicalAnalysis.out")
        morp_out_file      = open(morp_out_file_name, "w")

        lemma_list = []
        print "================================="
        print "Morphological Complexity Analysis"
        print "================================="
        for language_name, language_obj in self.analysed_lang_data.iteritems():

            lemma_list.append((language_name, language_obj.unique_word_type_to_lemma_ratio))

            #process each data line from the language file
            language_out_file = os.path.join(self.lang_out_dir, "%s.out" % language_obj.language_in_file.split('/')[-1])

            morp_text  = "Language          : %s\n" % language_obj.language_name
            morp_text += "Lemma             : %s\n" % language_obj.max_lemma
            morp_text += "Word Types        : %d\n" % language_obj.max_lemma_word_types
            morp_text += "Unique Word Types : %d\n" % language_obj.u_word_types
            morp_text += "Unique Lemma      : %d\n" % language_obj.u_lemmas
            morp_text += "Unique Word Type to Unique Lemma Ratio : %.10f\n\n" % language_obj.unique_word_type_to_lemma_ratio
            print morp_text
            morp_out_file.write(morp_text.encode('utf8'))
            self.data_parser.write_lemma_data(language_out_file, language_obj.sorted_data_list)
        morp_out_file.close()
        sorted_lemma_list = sorted(lemma_list, key=lambda x:x[1])
        print "\n".join(["%s %.10f" % (a, b) for a, b in sorted_lemma_list])
        print "Individual Language Analysis reports added to %s" % self.lang_out_dir

    def analyse_plots(self, rank=1, log_plot=False):
        out_file      = "Plots_Zipfian_Analysis_rank_%d.pdf" % rank
        if log_plot:
            out_file  = "Log_Plots_Zipfian_Analysis_rank_%d.pdf" % rank
        lang_out_file = os.path.join(self.lang_out_dir, out_file)
        pdf_file      = PdfPages(lang_out_file)

        print "==========================================="
        print "Plot Based analysis of zipfian distribution"
        print "==========================================="

        print "Considered ranks >= %d..." % rank
        if log_plot:
            print "Constructing plots with Log(word rank) in x-axis and Log(word frequency) in y-axis..."
        else:
            print "Constructing plots with word rank in x-axis and word frequency in y-axis..."


        for language_name, language_obj in self.analysed_lang_data.iteritems():

            #draw the plot for all words
            text_str   = "Rank/Frequency plot analysis [Language : %s, Rank >= %d]" % (language_name, rank)
            x_axis_str = "Word Rank"
            y_axis_str = "Word Frequency"
            if log_plot:
                text_str = "Log Rank/Log Frequency plot analysis [Language : %s, Rank >= %d]" % (language_name, rank)
                x_axis_str = "Log(Word Rank)"
                y_axis_str = "Log(Word Frequency)"
            plt.figure(text_str)
            if log_plot:
                lang_plot, = plt.plot(language_obj.log_rank_list[rank - 1:], language_obj.log_freq_list[rank - 1:], 'r')
            else:
                lang_plot, = plt.plot(language_obj.rank_list[rank - 1:], language_obj.freq_list[rank - 1:], 'r')
            if log_plot:
                plt.text(-1, -1, "Slope : %.10f" % language_obj.slope, ha='left')
                plt.text(-1, -1.5, "k       : %.10f" % -(language_obj.slope), ha="left")
            plt.xlabel(x_axis_str)
            plt.ylabel(y_axis_str)
            plt.title("Language : %s" % language_name)
            plt.savefig(pdf_file, format="pdf")
        print "Plots generated at %s" % lang_out_file
        pdf_file.close()

    def least_squares_analysis(self, use_lemma=False):
        new_list = []
        out_file      = "Least_Square_Regression_Word.pdf"
        if use_lemma:
            out_file  = "Least_Square_Regression_Lemma.pdf"
        lang_out_file = os.path.join(self.lang_out_dir, out_file)
        pdf_file      = PdfPages(lang_out_file)

        print "=============================================="
        print "Least Square Linear Regression Based Analysis"
        print "=============================================="
        if use_lemma:
            print "Using Lemma Frequency..."
            print "Constructing plots with log(lemma rank) in x-axis and log(lemma frequency) in y-axis"
        else:
            print "Using Word Frequency..."
            print "Constructing plots with log(word rank) in x-axis and log(word frequency) in y-axis"
            
        

        #Collect all slopes
        slopes_list = []
        for language_name, language_obj in self.analysed_lang_data.iteritems():
            if use_lemma:
                x_list = language_obj.log_lemma_rank_list
                y_list = language_obj.log_lemma_freq_list
            else:
                x_list = language_obj.log_rank_list
                y_list = language_obj.log_freq_list
            #form a matrix for x coordinates
            xmatrix          = np.array([x_list, np.ones(len(x_list))])
            #apply least square regression and find slope and intercept
            slope, intercept = np.linalg.lstsq(xmatrix.T, y_list)[0]
            slopes_list.append(slope)

            #get the ymatrix applying y = mx + c
            line = slope * np.array(x_list) + intercept

            #plot the figure
            if not use_lemma:
                figure_name = "Least Squares Regression Analysis with Word frequency [Language : %s]" % language_name
                x_axis_str  = "Log(Word Rank)"
                y_axis_str  = "Log(Word Frequency)"
            else:
                figure_name = "Least Squares Regression Analysis with Lemma frequency [Language : %s]" % language_name
                x_axis_str  = "Log(Lemma Rank)"
                y_axis_str  = "Log(Lemma Frequency)"
            plt.figure(figure_name)
            plt.plot(x_list, line, 'r-', x_list, y_list, 'b-')
            plt.xlabel(x_axis_str)
            plt.ylabel(y_axis_str)
            plt.title("Language : %s  [Slope : %.10f]" % (language_name, slope))
            plt.text(-1, -3, "Slope : %.10f" % language_obj.slope, ha='left')
            plt.text(-1, -3.5, "k       : %.10f" % -(language_obj.slope), ha="left")
            plt.legend(('fitted Line [By LstSqReg]', 'log-log plot'), 'upper center', shadow=True, fancybox=True)
            plt.savefig(pdf_file, format="pdf")
            new_list.append((language_name, -1*slope, abs(1+slope)))
        pdf_file.close()
        print "Plots generated at %s" % lang_out_file
        mean = np.average(slopes_list)
        std  = np.std(slopes_list)
        mean_k = np.average((-1*np.array(slopes_list)))
        std_k  = np.std((-1 * np.array(slopes_list)))
        print "List of Languages with k values"
        print "-------------------------------"
        print "\n".join(["%s %.10f %.10f" % (language_name, k , a) for language_name, k, a in sorted(new_list, key=lambda x:x[1], reverse=True)])
        print
        if use_lemma:
            print "Least Squares Regression Analysis using Lemma Frequency"
        else:
            print "Least Squares Regression Analysis using Word Frequency"
        print "\tMean of slopes               : %.10f" % mean
        print "\tStandard Deviation of slopes : %.10f" % std
        print "\tMean of k                    : %.10f" % mean_k
        print "\tStandard Deviation of k      : %.10f" % std_k



    def terminate_with_error(self):
        print "Error : Check the log file for info"
        sys.exit(2)


if __name__ == "__main__":
    logger         = initialize_logger(os.path.splitext(sys.argv[0])[0]+ os.path.extsep + "log")
    lang_dict      = {
                        'Bulgarian' : "hw2-data/orwell-bg.txt",\
                        'Czech'     : "hw2-data/orwell-cs.txt",\
                        'English'   : "hw2-data/orwell-en.txt",\
                        'Estonian'  : "hw2-data/orwell-et.txt",\
                        'Farsi'     : "hw2-data/orwell-fa.txt",\
                        'Hungarian' : "hw2-data/orwell-hu.txt",\
                        'Polish'    : "hw2-data/orwell-pl.txt",\
                        'Romanian'  : "hw2-data/orwell-ro.txt",\
                        'Slovak'    : "hw2-data/orwell-sk.txt",\
                        'Slovene'   : "hw2-data/orwell-sl.txt",\
                        'Serbian'   : "hw2-data/orwell-sr.txt"\
                     }
    lang_analyzer  = LanguageAnalyzer(logger, lang_dict)
    lang_analyzer.analyze_morphology()
    lang_analyzer.analyse_plots()
    lang_analyzer.analyse_plots(rank=300)
    lang_analyzer.analyse_plots(log_plot=True)
    lang_analyzer.least_squares_analysis()
    lang_analyzer.least_squares_analysis(use_lemma=True)

