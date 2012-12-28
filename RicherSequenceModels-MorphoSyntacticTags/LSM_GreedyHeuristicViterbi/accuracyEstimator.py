from viterbi import Viterbi

class AccuracyEstimator:
    def __init__(self, logger, data_parser, start_tag="START", stop_tag="STOP"):
        self.logger                     = logger
        self.data_parser                = data_parser
        self.start_tag                  = start_tag
        self.stop_tag                   = stop_tag
        self.word_count                 = 0
        self.unseen_word_count          = 0
        self.correctly_tagged           = 0
        self.unseen_correctly_tagged    = 0
        self.total_accuracy             = 0.0
        self.unseen_accuracy            = 0.0
        self.tags_blacklist             = ["START", "STOP"]


    def compute_parameters(self, tagging_model, test_file, out_file, lang_name, start_line=5501):
        if not tagging_model:
            self.logger.info("Tagging model not found")
            return self.total_accuracy, self.unseen_accuracy

        self.logger.info("Computing parameters using test file %s from line no : %d" % (test_file, start_line))
        o_ptr = open(out_file, "w")
        for line_no, word_list in self.data_parser.next(test_file):
            if line_no < start_line:
                continue

            word_list = [(word, tag) for word, tag in word_list if tag not in self.tags_blacklist]

            predicted_tags = self.predict_tags(word_list, tagging_model)

            for index, (word, tag) in enumerate(word_list):

                predicted_tag = predicted_tags[index]

                #total accuracy calculation
                if predicted_tag == tag:
                    self.correctly_tagged += 1
                else:
                    o_line = "%s#%d: %s\t%s\t%s\n" % (lang_name, line_no, word, tag, predicted_tag)
                    try:
                        o_ptr.write(o_line.encode("utf8"))
                    except:
                        o_ptr.write(o_line)
                self.word_count += 1

                #unseen accuracy calculation
                if tagging_model.is_unseen(word):
                    if predicted_tag == tag:
                        self.unseen_correctly_tagged += 1
                    self.unseen_word_count += 1

        o_ptr.close()
        self.total_accuracy     = ( (self.correctly_tagged * 1.0) / self.word_count ) * 100.0
        self.unseen_accuracy    = ( (self.unseen_correctly_tagged * 1.0) / self.unseen_word_count ) * 100.0
        self.logger.info("************************************************************************************")
        self.logger.info("CORRECTLY TAGGED              : %d" % self.correctly_tagged)
        self.logger.info("WORD COUNT                    : %d" % self.word_count)
        self.logger.info("UNSEEN CORRECTLY TAGGED       : %d" % self.unseen_correctly_tagged)
        self.logger.info("UNSEEN WORD COUNT             : %d" % self.unseen_word_count)
        self.logger.info("TOTAL ACCURACY                : %.10f" % self.total_accuracy)
        self.logger.info("UNSEEN ACCURACY               : %.10f" % self.unseen_accuracy)
        self.logger.info("************************************************************************************")
        return self.total_accuracy, self.unseen_accuracy

    def predict_tags(self, word_list, tagging_model):
        if isinstance(tagging_model, Viterbi):
            return tagging_model.predict_sequence([word for word , tag in word_list])[1]
        return [tagging_model.predict(word) for word, tag in word_list]

    def reset(self):
        self.word_count                 = 0
        self.unseen_word_count          = 0
        self.correctly_tagged           = 0
        self.unseen_correctly_tagged    = 0
        self.total_accuracy             = 0.0
        self.unseen_accuracy            = 0.0


