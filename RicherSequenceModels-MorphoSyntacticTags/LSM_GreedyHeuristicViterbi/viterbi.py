class Viterbi:
    def __init__(self, logger, hmm_obj):
        self.logger     = logger
        self.hmm_obj    = hmm_obj
        self.tag_list   = []
        self.trans_info = {}
        self.special_tags = ["START", "STOP"]

    def train(self, training_file, end_line=5500):
        self.hmm_obj.train(training_file, end_line)
        self.tag_list = self.hmm_obj.hidden_states
        self.trans_info = self.hmm_obj.trans_info

    def is_unseen(self, word):
        return self.hmm_obj.is_unseen(word)

    def reset(self):
        self.tag_list = []
        self.trans_info = {}
        self.hmm_obj.reset()


    def predict_sequence(self, word_list):
        num_words = len(word_list)
        num_tags  = len(self.tag_list)
        result_list = []

        tag_rank = {}
        for index, tag in enumerate(self.tag_list):
            tag_rank[tag] = index

        for i in xrange(num_words + 1):
            result_list.append([[-10000000, None] for j in xrange(num_tags + 1)])

        for t in xrange(num_words):
            for j in xrange(num_tags):
                cand_tags = self.trans_info.get(self.tag_list[j], set())
                for c_tag in cand_tags:
                    i = tag_rank[c_tag]
                    result = self.hmm_obj.new_compute_prev(result_list, t - 1, i, j, word_list, self.tag_list)
                    if result > result_list[t][j][0]:
                        if t - 1 < 0:
                            result_list[t][j] = [result, [self.tag_list[j]]]
                        else:
                            tag_elts = []
                            if result_list[t - 1][i][1] is None:
                                continue
                            for elt in result_list[t - 1][i][1]:
                                tag_elts.append(elt)
                            tag_elts.append(self.tag_list[j])
                            result_list[t][j] = [result, tag_elts]


        #compute final info
        for i in xrange(num_tags):
            result = max(result_list[num_words][num_tags][0], self.hmm_obj.new_compute_final(result_list, i, num_words, self.tag_list))
            if result > result_list[num_words][num_tags][0]:
                tag_elts = []
                for elt in result_list[t][i][1]:
                    tag_elts.append(elt)
                result_list[num_words][num_tags] = [result, tag_elts]

        return result_list[num_words][num_tags]

    def predict_sequence_old(self, word_list):
        #self.tag_list    = ['START', 'C', 'P', 'S', 'D', 'A', 'V', 'N', 'PUN', 'M', 'STOP', 'R', 'X', 'I']
        #self.tag_list    = ["PUN", "N", "D", "I", "M", "C", "S", "A", "P", "R", "V", "X", "START", "STOP"]
        num_words        = len(word_list)
        num_tags         = len(self.tag_list)
        result_list      = []


        #initialize to zeros
        for i in xrange(num_words + 1):
            result_list.append([-1000000000, None])


        #compute table info
        for t in xrange(num_words):
            for j in xrange(num_tags):
                    result = self.hmm_obj.compute_prev(result_list, t, j, word_list, self.tag_list)
                    if result_list[t][1] is None or result > result_list[t][0]:
                        result_list[t] = [result, self.tag_list[j]]


        return [tag for prob, tag in result_list]

