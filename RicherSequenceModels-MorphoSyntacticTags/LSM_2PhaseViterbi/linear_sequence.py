from viterbi import Viterbi

class LinearSequence:
    def __init__(self, logger, data_parser, use_avg=False, use_suffix=False, training_level=5, start_tag="START", stop_tag="STOP"):
        self.logger         = logger
        self.data_parser    = data_parser
        self.start_tag      = start_tag
        self.stop_tag       = stop_tag
        self.training_level = training_level
        self.tag_features   = set()
        self.word_features  = set()
        self.weights        = {}
        self.seen_words     = set()
        self.viterbi_obj    = Viterbi(logger, self)
        self.KEY_TAG        = "TAG_FEATURE"
        self.KEY_WORD       = "WORD_FEATURE"
        self.KEY_SUFFIX     = "SUFFIX_FEATURE"
        self.special_tags   = [start_tag, stop_tag]
        self.hidden_states  = []
        self.avg_weights    = {}
        self.use_avg        = use_avg
        self.use_suffix     = use_suffix
        self.trained        = False
        self.suffix_features= set()

    def reset(self):
        self.tag_features   = set()
        self.word_features  = set()
        self.weights        = {}
        self.seen_words     = set()
        self.trained        = False
        self.avg_weights    = {}
        self.hidden_states  = []
        self.suffix_features=set()

    def is_unseen(self, word):
        if word in self.seen_words:
            return False
        return True

    def train(self, training_file, end_line=5500):
        self.logger.info("Started training data from %s upto line %d" %(training_file, end_line))
        tags_info = {}
        for line_no, word_list in self.data_parser.next(training_file):

            if line_no > end_line:
                break

            prev_tag = None
            for index, (word, tag) in enumerate(word_list):
                #create feature space
                if prev_tag is not None:
                    self.tag_features.add((prev_tag, tag))
                self.word_features.add((tag, word))

                #suffix features
                if len(word) > 1:
                    self.suffix_features.add((word[-1:], tag))
                    if len(word) > 2:
                        self.suffix_features.add((word[-2:], tag))
                        if len(word) > 3:
                            self.suffix_features.add((word[-3:], tag))

                prev_tag = tag

                #if tag not in self.hidden_states:
                #    self.hidden_states.append(tag)
                tags_info[tag] = tags_info.setdefault(tag, 0) + 1

                self.seen_words.add(word)
        #self.viterbi_obj.tag_list = [tag for tag, count in sorted(tags_info.iteritems(), key=lambda x:x[1])]
        #self.hidden_states = [tag for tag, count in sorted(tags_info.iteritems(), key=lambda x:x[1])]
        self.viterbi_obj.tag_list = tags_info.keys()
        self.hidden_states        = tags_info.keys()
        print self.hidden_states
        self.logger.info("Completed parsing the training data to form feature space")
        self.logger.info("Tag features : %d" % len(self.tag_features))
        self.logger.info("Word features : %d" % len(self.word_features))
        self.logger.info("Suffix features : %d" % len(self.suffix_features))
        self.logger.info("Hidden States  : %d" % len(self.hidden_states))
        self.estimate_weights(training_file, end_line)
        self.trained = True

    def get_suffix_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights
        wt = 0
        if len(word) > 1:
            wt += weights.get(self.KEY_SUFFIX, {}).get((word[-1:], tag), 0)
            if len(word) > 2:
                wt += weights.get(self.KEY_SUFFIX, {}).get((word[-2:], tag), 0)
                if len(word) > 3:
                    wt += weights.get(self.KEY_SUFFIX, {}).get((word[-3:], tag), 0)
        return wt

    def get_cand_suffix_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights
        wt = 0
        if len(word) > 1:
            wt += weights.get("CAND_SUF", {}).get((word[-1:], tag), 0)
            if len(word) > 2:
                wt += weights.get("CAND_SUF", {}).get((word[-2:], tag), 0)
                if len(word) > 3:
                    wt += weights.get("CAND_SUF", {}).get((word[-3:], tag), 0)
        return wt

    def get_transition_feature(self, prev_tag, next_tag=None):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights
        if next_tag is None:
            next_tag = self.stop_tag

        return weights.get(self.KEY_TAG, {}).get((prev_tag, next_tag), 0)

    def get_cand_transition_feature(self, prev_tag, next_tag=None):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights
        if next_tag is None:
            next_tag = self.stop_tag

        return weights.get("CAND_TRA", {}).get((prev_tag, next_tag), 0)



    def get_emission_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        return weights.get(self.KEY_WORD, {}).get((tag, word), 0)

    def get_cand_emission_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        return weights.get("CAND_EMI", {}).get((tag, word), 0)



    def main_compute_prev(self, result_list, t, i, j, word_list, tag_list):
        addn_wt = 0
        if self.use_suffix:
            addn_wt = self.get_suffix_feature(tag_list[j], word_list[t + 1])

        if t < 0:
            if tag_list[i] == self.start_tag:
                return self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt
            else:
                return -10000000 + self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt

        return result_list[t][i][0] + self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt

    def main_compute_final(self, result_list, i, num_words, tag_list):
        return result_list[num_words - 1][i][0] + self.get_transition_feature(tag_list[i])

    def compute_final(self, result_hash, final, num_words):
        return result_hash[num_words - 1][final][0] + self.get_cand_transition_feature(final)

    def compute_prev(self, result_hash, t, word_list, cur, prev):
        addn_wt = self.get_cand_suffix_feature(cur, word_list[t])

        if t <= 0:
            return self.get_cand_transition_feature(prev, cur) + self.get_cand_emission_feature(cur, word_list[t]) + addn_wt

        return result_hash[t - 1][prev][0] + self.get_cand_transition_feature(prev, cur) + self.get_cand_emission_feature(cur, word_list[t]) + addn_wt

    def estimate_weights(self, training_file, end_line=5500):
        self.logger.info("Started estimating weights...")
        parse_level = 0
        multiplier = self.training_level * end_line
        for r_level in xrange(self.training_level):
            parse_level += 1
            for line_no, word_list in self.data_parser.next(training_file):

                if line_no > end_line:
                    break

                if line_no % 500 == 0:
                    print "LEVEL: %d : Processed %d lines..." % (r_level, line_no)

                new_word_list = [(word, tag) for word, tag in word_list if tag not in self.special_tags]
                predicted_tags  = self.viterbi_obj.predict_sequence([word for word, tag in new_word_list])
                self.reestimate_weights(new_word_list, predicted_tags, multiplier)

                multiplier -= 1

            self.logger.info("Completed parsing %d time(s) for estimating weights" % parse_level)
        self.logger.info("Completed estimating weights")
        self.logger.info("TAG WEIGHTS  : %d" % len(self.weights.get(self.KEY_TAG, {})))
        self.logger.info("WORD WEIGHTS  : %d" % len(self.weights.get(self.KEY_WORD, {})))


    def reestimate_weights(self, word_list, predicted_tags, multiplier):
        prev_tag        = self.start_tag
        pred_prev_tag   = self.start_tag
        prev_main_tag        = self.start_tag
        pred_prev_main_tag   = self.start_tag
        local_diff      = {}
        for index, (word, tag) in enumerate(word_list):
            if tag in self.special_tags:
                pred_tag = tag
            else:
                if predicted_tags:
                    pred_tag = predicted_tags[index]
                else:
                    pred_tag = None

            if tag in ["PUN", "START", "STOP"]:
                main_tag = tag
            else:
                main_tag = tag[0]

            if pred_tag in ["PUN", "START", "STOP"]:
                pred_main_tag = pred_tag
            else:
                pred_main_tag = pred_tag[0]


            #weights of tags
            if prev_main_tag is not None:
                #count = self.weights.setdefault(self.KEY_TAG, {}).setdefault((prev_tag, tag), 0)
                #self.weights[self.KEY_TAG][(prev_tag, tag)] = count + 1
                count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((prev_main_tag, main_tag), 0)
                local_diff[self.KEY_TAG][(prev_main_tag, main_tag)] = count + 1
                count = local_diff.setdefault("CAND_TRA", {}).setdefault((prev_tag, tag), 0)
                local_diff["CAND_TRA"][(prev_tag, tag)] = count + 1



            if predicted_tags and pred_prev_main_tag is not None:
                #count = self.weights.setdefault(self.KEY_TAG, {}).setdefault((pred_prev_tag, pred_tag), 0)
                #self.weights[self.KEY_TAG][(pred_prev_tag, pred_tag)] = count - 1
                count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((pred_prev_main_tag, pred_main_tag), 0)
                local_diff[self.KEY_TAG][(pred_prev_main_tag, pred_main_tag)] = count - 1
                count = local_diff.setdefault("CAND_TRA", {}).setdefault((pred_prev_tag, pred_tag), 0)
                local_diff["CAND_TRA"][(pred_prev_tag, pred_tag)] = count - 1


            #weights of words
            if tag not in self.special_tags:
                #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((tag, word), 0)
                #self.weights[self.KEY_WORD][(tag, word)] = count + 1
                count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((main_tag, word), 0)
                local_diff[self.KEY_WORD][(main_tag, word)] = count + 1

                count = local_diff.setdefault("CAND_EMI", {}).setdefault((tag, word), 0)
                local_diff["CAND_EMI"][(tag, word)] = count + 1


                if predicted_tags:
                    #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((pred_tag, word), 0)
                    #self.weights[self.KEY_WORD][(pred_tag, word)] = count - 1
                    count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((pred_main_tag, word), 0)
                    local_diff[self.KEY_WORD][(pred_main_tag, word)] = count - 1

                    count = local_diff.setdefault("CAND_EMI", {}).setdefault((pred_tag, word), 0)
                    local_diff["CAND_EMI"][(pred_tag, word)] = count - 1

            #weight of suffix
            if len(word) > 1:
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], main_tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], main_tag)] = count + 1
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], pred_main_tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], pred_main_tag)] = count - 1
                count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-1:], tag), 0)
                local_diff["CAND_SUF"][(word[-1:], tag)] = count + 1
                count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-1:], pred_tag), 0)
                local_diff["CAND_SUF"][(word[-1:], pred_tag)] = count - 1
                if len(word) > 2:
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], main_tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], main_tag)] = count + 1
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], pred_main_tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], pred_main_tag)] = count - 1
                    count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-2:], tag), 0)
                    local_diff["CAND_SUF"][(word[-2:], tag)] = count + 1
                    count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-2:], pred_tag), 0)
                    local_diff["CAND_SUF"][(word[-2:], pred_tag)] = count - 1
                    if len(word) > 3:
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], main_tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], main_tag)] = count + 1
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], pred_main_tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], pred_main_tag)] = count - 1
                        count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-3:], tag), 0)
                        local_diff["CAND_SUF"][(word[-3:], tag)] = count + 1
                        count = local_diff.setdefault("CAND_SUF", {}).setdefault((word[-3:], pred_tag), 0)
                        local_diff["CAND_SUF"][(word[-3:], pred_tag)] = count - 1


            prev_tag        = tag
            prev_main_tag   = main_tag
            pred_prev_tag   = pred_tag
            pred_prev_main_tag = pred_main_tag

        count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((main_tag, self.stop_tag), 0)
        local_diff[self.KEY_TAG][(main_tag, self.stop_tag)] = count + 1
        count = local_diff.setdefault("CAND_TRA", {}).setdefault((tag, self.stop_tag), 0)
        local_diff["CAND_TRA"][(tag, self.stop_tag)] = count + 1

        count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((pred_main_tag, self.stop_tag), 0)
        local_diff[self.KEY_TAG][(pred_main_tag, self.stop_tag)] = count - 1
        count = local_diff.setdefault("CAND_TRA", {}).setdefault((pred_tag, self.stop_tag), 0)
        local_diff["CAND_TRA"][(pred_tag, self.stop_tag)] = count - 1


        for tag_type, info_hash in local_diff.iteritems():
            for key, value in info_hash.iteritems():
                count = self.weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.weights[tag_type][key] = count + value
                wt = self.avg_weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.avg_weights[tag_type][key] = wt + multiplier * value


