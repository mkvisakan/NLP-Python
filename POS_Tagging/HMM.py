import gmpy2

class HMM:
    def __init__(self, logger, data_parser, skew_unseen=False, start_tag="START", stop_tag="STOP"):
        self.logger             = logger
        self.data_parser        = data_parser
        self.start_tag          = start_tag
        self.stop_tag           = stop_tag
        self.transition_info    = {}
        self.emission_info      = {}
        self.tag_info           = {}
        self.special_tags       = [start_tag, stop_tag]
        self.word_types         = set([])
        self.vocab_count        = 0
        self.hidden_states      = []
        self.seen_words         = set()
        self.frequent_tag       = None
        self.skew_unseen        = skew_unseen

    def train(self, training_file, end_line=5500):
        self.logger.info("Started training data from %s upto line %d" % (training_file, end_line))
        for line_no, word_list in self.data_parser.next(training_file):
            
            self.word_types.update([word for word, tag in word_list if word not in self.special_tags])
            
            if line_no > end_line:
                break

            for index, (word, tag) in enumerate(word_list):
                tag_count           = self.tag_info.setdefault(tag, 0)
                self.tag_info[tag]  = tag_count + 1
                
                if tag == self.start_tag:
                    continue

                self.seen_words.add(word)

                tag_info        = self.transition_info.setdefault(word_list[index - 1][1], {})
                tag_count       = tag_info.setdefault(tag, 0)
                tag_info[tag]   = tag_count + 1

                word_info       = self.emission_info.setdefault(tag, {})
                word_count      = word_info.setdefault(word, 0)
                word_info[word] = word_count + 1
        self.vocab_count    =  len(self.word_types)
        self.hidden_states  = self.tag_info.keys()
        self.frequent_tag   = sorted(self.tag_info.iteritems(), key=lambda x:x[1], reverse=True)[0][0]
        self.logger.info("Completed training for HMM...")
        self.logger.info("VOCAB COUNT : %d" % self.vocab_count)
        self.logger.info("FREQUENT TAG : %s" % self.frequent_tag)
        
    def is_unseen(self, word):
        if word in self.seen_words:
            return False
        return True
        

    def reset(self):
        self.logger.info("Resetting HMM parameters")
        self.transition_info = {}
        self.emission_info   = {}
        self.tag_info        = {}
        self.word_types      = set([])
        self.vocab_count     = 0
        self.seen_words      = set()
        self.frequent_tag    = None
        
    def get_transition_prob(self, prev_tag, next_tag=None):
        if next_tag is None:
            next_tag = self.stop_tag

        numerator   = self.transition_info.get(prev_tag, {}).get(next_tag, 0)
        denominator = self.tag_info.get(prev_tag, 0)

        return (numerator) / (denominator * 1.0)

    def get_emission_prob(self, tag, word, scaling_factor=0.000001):
        word_count = self.emission_info.get(tag, {}).get(word, 0)

        if self.skew_unseen and tag == self.frequent_tag:
            numerator   = word_count + (1000 * scaling_factor)
            denominator = self.tag_info.get(tag, 0) + (self.vocab_count + (1000 * scaling_factor))
        else:
            numerator   = word_count + scaling_factor
            denominator = self.tag_info.get(tag, 0) + (self.vocab_count * scaling_factor)

        return (numerator) / (denominator * 1.0)
        
        
                
    def compute_prev(self, result_list, t, i, j, word_list, tag_list):
        if t < 0:
            return gmpy2.log2(self.get_transition_prob(tag_list[i], tag_list[j])) + gmpy2.log2(self.get_emission_prob(tag_list[j], word_list[t + 1]))
        return result_list[t][i][0] + gmpy2.log2(self.get_transition_prob(tag_list[i], tag_list[j])) + gmpy2.log2(self.get_emission_prob(tag_list[j], word_list[t + 1]))

    def compute_final(self, result_list, i, num_words, tag_list):
        return result_list[num_words - 1][i][0] + gmpy2.log2(self.get_transition_prob(tag_list[i]))
