#
#Author       : Kumaresh Visakan Murugan
#Email        : mvisakan@cs.wisc.edu
#Class        : CS545 - NLP
#Instructor   : Prof. Benjamin Snyder
#

import codecs
import os
from loggerUtils import initialize_logger

class Schema:
    def __init__(self):
        self.token_id, self.word_token,\
        self.m_analysis, self.lemma      = range(4)
        self.language_schema             = dict(token_id=self.token_id, word_token=self.word_token, m_analysis=self.m_analysis, lemma=self.lemma)

    def get_schema_len(self):
        return len(self.language_schema)

    def get_word_type(self, data_list):
        return data_list[self.word_token]

    def get_lemma(self, data_list):
        return data_list[self.lemma]

class DataParser:
    def __init__(self, logger):
        self.logger    = logger


    def iterate(self, data_file, encoding="utf-8", tab_spaces=3):
        self.logger.info("Started reading data from %s" % data_file)
        #check file existence
        if not os.path.exists(data_file):
            self.logger.info("Data file %s not found !!!" % data_file)
            return
        #open file
        file_reader = codecs.open(data_file, "r", encoding)

        #iterate over lines using file_reader
        line = file_reader.readline()
        while line:

            #counter logging
            counter = 1
            if counter % 5000 == 0:
                self.logger.info("Processed %s lines" % counter)

            #yield data
            data_list = line.strip().split('\t', tab_spaces)
            yield data_list
            line = file_reader.readline()

        #close the file
        self.logger.info("Completed reading data from %s" % data_file)
        file_reader.close()

    def write_lemma_data(self, out_file, data_list, encoding="utf-8"):
        self.logger.info("Writing Lemma data to output file %s" % out_file)
        file_writer = codecs.open(out_file, "w", encoding)
        for lemma, word_types in data_list:
            file_writer.write("%s\t%d\n" % (lemma, len(word_types)))
        self.logger.info("Completed writing data to %s" % out_file)
        file_writer.close()


if __name__ == "__main__":
    logger  = initialize_logger()
    obj     = DataParser(logger)
    count   = 0
    for line in obj.iterate("hw2-data/orwell-en.txt"):
        count += 1
        print line
        if count == 10:
            break
