import gmpy2

class Viterbi:
    def __init__(self, logger, hmm_obj):
        self.logger     = logger
        self.hmm_obj    = hmm_obj
        self.tag_list   = []
        self.special_tags = ["START", "STOP"]

    def train(self, training_file, end_line=5500):
        self.hmm_obj.train(training_file, end_line)
        self.tag_list = self.hmm_obj.hidden_states

    def is_unseen(self, word):
        return self.hmm_obj.is_unseen(word)

    def reset(self):
        self.tag_list = []
        self.hmm_obj.reset()

    def predict_sequence(self, word_list):
        #self.tag_list    = ['START', 'C', 'P', 'S', 'D', 'A', 'V', 'N', 'PUN', 'M', 'STOP', 'R', 'X', 'I']
        #self.tag_list    = ["PUN", "N", "D", "I", "M", "C", "S", "A", "P", "R", "V", "X", "START", "STOP"]
        num_words        = len(word_list)
        num_tags         = len(self.tag_list)
        result_list      = []
        

        #initialize to zeros
        for i in xrange(num_words + 1):
            result_list.append([[-1000000000, []] for j in xrange(num_tags + 1)])

        
        #compute table info
        for t in xrange(num_words):
            for j in xrange(num_tags):
                for i in xrange(num_tags):
                    result = self.hmm_obj.compute_prev(result_list, t - 1, i, j, word_list, self.tag_list)
                    if result > result_list[t][j][0]:
                        if t - 1 < 0:
                            result_list[t][j] = [result, [self.tag_list[j]]]
                        else:
                            tag_elts = []
                            for elt in result_list[t - 1][i][1]:
                                tag_elts.append(elt)
                            tag_elts.append(self.tag_list[j])
                            result_list[t][j] = [result, tag_elts]
            

        #compute final info
        for i in xrange(num_tags):
            result = max(result_list[num_words][num_tags][0], self.hmm_obj.compute_final(result_list, i, num_words, self.tag_list))
            if result > result_list[num_words][num_tags][0]:
                tag_elts = []
                for elt in result_list[t][i][1]:
                    tag_elts.append(elt)
                result_list[num_words][num_tags] = [result, tag_elts]

        return result_list[num_words][num_tags]
            
