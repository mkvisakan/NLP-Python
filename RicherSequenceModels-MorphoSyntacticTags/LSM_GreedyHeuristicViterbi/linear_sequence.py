from viterbi import Viterbi

class LinearSequence:
    def __init__(self, logger, data_parser, use_avg=True, use_suffix=True, training_level=5, start_tag="START", stop_tag="STOP"):
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
        self.trans_info = {}

    def reset(self):
        self.tag_features   = set()
        self.word_features  = set()
        self.weights        = {}
        self.seen_words     = set()
        self.trained        = False
        self.avg_weights    = {}
        self.hidden_states  = []
        self.suffix_features=set()
        self.trans_info = {}

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
                if prev_tag:
                    result_set = self.trans_info.setdefault(tag, set())
                    result_set.add(prev_tag)
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
        self.viterbi_obj.trans_info = self.trans_info
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


    def get_emission_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        return weights.get(self.KEY_WORD, {}).get((tag, word), 0)

    def get_prev_word_feature(self, tag, cur_word, prev_word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        return weights.get("PREV_WORD", {}).get((tag, cur_word, prev_word), 0)

    def get_elder_word_feature(self, tag, cur_word, prev_word, elder_word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        return weights.get("ELDER_WORD", {}).get((tag, cur_word, prev_word, elder_word), 0)

    def get_hypen_feature(self, tag, word):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights

        if "-" in word:
            return weights.get("HYPEN", {}).get(tag, 0)
        else:
            return 0


    def compute_prev(self, result_list, t, j, word_list, tag_list):
        addn_wt = 0
        if self.use_suffix:
            addn_wt = self.get_suffix_feature(tag_list[j], word_list[t])

        if t <= 0:
            addn_wt += self.get_prev_word_feature(tag_list[j], word_list[t], None)
        else:
            addn_wt += self.get_prev_word_feature(tag_list[j], word_list[t], word_list[t-1])

        if t == 0:
            addn_wt += self.get_elder_word_feature(tag_list[j], word_list[t], None, None)
        elif t == 1:
            addn_wt += self.get_elder_word_feature(tag_list[j], word_list[t], word_list[t-1], None)
        else:
            addn_wt += self.get_elder_word_feature(tag_list[j], word_list[t], word_list[t-1], word_list[t-2])

        addn_wt += self.get_hypen_feature(tag_list[j], word_list[t])

        return self.get_emission_feature(tag_list[j], word_list[t]) + addn_wt

    def get_transition_feature(self, prev_tag, next_tag=None):
        if self.trained and self.use_avg:
            weights = self.avg_weights
        else:
            weights = self.weights
        if next_tag is None:
            next_tag = self.stop_tag

        return weights.get(self.KEY_TAG, {}).get((prev_tag, next_tag), 0)

    def new_compute_prev(self, result_list, t, i, j, word_list, tag_list):
        addn_wt = 0
        if self.use_suffix:
            addn_wt = self.get_suffix_feature(tag_list[j], word_list[t + 1])

        if t < 0:
            if tag_list[i] == self.start_tag:
                return self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt
            else:
                return -10000000 + self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt

        return result_list[t][i][0] + self.get_transition_feature(tag_list[i], tag_list[j]) + self.get_emission_feature(tag_list[j], word_list[t + 1]) + addn_wt

    def new_compute_final(self, result_list, i, num_words, tag_list):
        return result_list[num_words - 1][i][0] + self.get_transition_feature(tag_list[i])

    def estimate_weights(self, training_file, end_line=5500):
        self.logger.info("Started estimating weights...")
        parse_level = 0
        multiplier = self.training_level * end_line
        for r_level in xrange(self.training_level):
            parse_level += 1
            for line_no, word_list in self.data_parser.next(training_file):

                if line_no > end_line:
                    break

                if line_no % 100 == 0:
                    print "LEVEL %d : processed %d lines" % (r_level, line_no)

                new_word_list = [(word, tag) for word, tag in word_list if tag not in self.special_tags]
                predicted_tags  = self.viterbi_obj.predict_sequence([word for word, tag in new_word_list])[1]

                self.new_reestimate_weights(new_word_list, predicted_tags, multiplier)

                multiplier -= 1

            self.logger.info("Completed parsing %d time(s) for estimating weights" % parse_level)
        self.logger.info("Completed estimating weights")
        self.logger.info("TAG WEIGHTS  : %d" % len(self.weights.get(self.KEY_TAG, {})))
        self.logger.info("WORD WEIGHTS  : %d" % len(self.weights.get(self.KEY_WORD, {})))

    def new_reestimate_weights(self, word_list, predicted_tags, multiplier):
        prev_tag        = self.start_tag
        pred_prev_tag   = self.start_tag
        local_diff      = {}
        for index, (word, tag) in enumerate(word_list):
            if tag in self.special_tags:
                pred_tag = tag
            else:
                if predicted_tags:
                    pred_tag = predicted_tags[index]
                else:
                    pred_tag = None


            #weights of tags
            if prev_tag is not None:
                #count = self.weights.setdefault(self.KEY_TAG, {}).setdefault((prev_tag, tag), 0)
                #self.weights[self.KEY_TAG][(prev_tag, tag)] = count + 1
                count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((prev_tag, tag), 0)
                local_diff[self.KEY_TAG][(prev_tag, tag)] = count + 1


            if predicted_tags and pred_prev_tag is not None:
                #count = self.weights.setdefault(self.KEY_TAG, {}).setdefault((pred_prev_tag, pred_tag), 0)
                #self.weights[self.KEY_TAG][(pred_prev_tag, pred_tag)] = count - 1
                count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((pred_prev_tag, pred_tag), 0)
                local_diff[self.KEY_TAG][(pred_prev_tag, pred_tag)] = count - 1


            #weights of words
            if tag not in self.special_tags:
                #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((tag, word), 0)
                #self.weights[self.KEY_WORD][(tag, word)] = count + 1
                count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((tag, word), 0)
                local_diff[self.KEY_WORD][(tag, word)] = count + 1


                if predicted_tags:
                    #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((pred_tag, word), 0)
                    #self.weights[self.KEY_WORD][(pred_tag, word)] = count - 1
                    count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((pred_tag, word), 0)
                    local_diff[self.KEY_WORD][(pred_tag, word)] = count - 1

            #weight of suffix
            if len(word) > 1:
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], tag)] = count + 1
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], pred_tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], pred_tag)] = count - 1
                if len(word) > 2:
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], tag)] = count + 1
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], pred_tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], pred_tag)] = count - 1
                    if len(word) > 3:
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], tag)] = count + 1
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], pred_tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], pred_tag)] = count - 1


            prev_tag        = tag
            pred_prev_tag   = pred_tag

        count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((tag, self.stop_tag), 0)
        local_diff[self.KEY_TAG][(tag, self.stop_tag)] = count + 1

        count = local_diff.setdefault(self.KEY_TAG, {}).setdefault((pred_tag, self.stop_tag), 0)
        local_diff[self.KEY_TAG][(pred_tag, self.stop_tag)] = count - 1


        for tag_type, info_hash in local_diff.iteritems():
            for key, value in info_hash.iteritems():
                count = self.weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.weights[tag_type][key] = count + value
                wt = self.avg_weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.avg_weights[tag_type][key] = wt + multiplier * value


    def reestimate_weights(self, word_list, predicted_tags, multiplier):
        prev_tag        = self.start_tag
        pred_prev_tag   = self.start_tag
        prev_word       = None
        local_diff      = {}
        for index, (word, tag) in enumerate(word_list):
            if tag in self.special_tags:
                pred_tag = tag
            else:
                if predicted_tags:
                    pred_tag = predicted_tags[index]
                else:
                    pred_tag = None

            #weights of words
            if tag not in self.special_tags:
                #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((tag, word), 0)
                #self.weights[self.KEY_WORD][(tag, word)] = count + 1
                count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((tag, word), 0)
                local_diff[self.KEY_WORD][(tag, word)] = count + 1


                if predicted_tags:
                    #count = self.weights.setdefault(self.KEY_WORD, {}).setdefault((pred_tag, word), 0)
                    #self.weights[self.KEY_WORD][(pred_tag, word)] = count - 1
                    count = local_diff.setdefault(self.KEY_WORD, {}).setdefault((pred_tag, word), 0)
                    local_diff[self.KEY_WORD][(pred_tag, word)] = count - 1

            #weight of suffix
            if len(word) > 1:
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], tag)] = count + 1
                count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-1:], pred_tag), 0)
                local_diff[self.KEY_SUFFIX][(word[-1:], pred_tag)] = count - 1
                if len(word) > 2:
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], tag)] = count + 1
                    count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-2:], pred_tag), 0)
                    local_diff[self.KEY_SUFFIX][(word[-2:], pred_tag)] = count - 1
                    if len(word) > 3:
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], tag)] = count + 1
                        count = local_diff.setdefault(self.KEY_SUFFIX, {}).setdefault((word[-3:], pred_tag), 0)
                        local_diff[self.KEY_SUFFIX][(word[-3:], pred_tag)] = count - 1

            #prev word feature
            count = local_diff.setdefault("PREV_WORD", {}).setdefault((tag, word, prev_word), 0)
            local_diff["PREV_WORD"][(tag, word, prev_word)] = count + 1


            count = local_diff.setdefault("PREV_WORD", {}).setdefault((pred_tag, word, prev_word), 0)
            local_diff["PREV_WORD"][(pred_tag, word, prev_word)] = count - 1

            #elder word
            if index == 0:
                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((tag, word, None, None), 0)
                local_diff["ELDER_WORD"][(tag, word, None, None)] = count + 1

                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((pred_tag, word, None, None), 0)
                local_diff["ELDER_WORD"][(pred_tag, word, None, None)] = count - 1
            elif index == 1:
                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((tag, word, word_list[index-1], None), 0)
                local_diff["ELDER_WORD"][(tag, word, word_list[index-1], None)] = count + 1

                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((pred_tag, word, word_list[index-1], None), 0)
                local_diff["ELDER_WORD"][(pred_tag, word, word_list[index-1], None)] = count - 1
            else:
                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((tag, word, word_list[index-1], word_list[index-2]), 0)
                local_diff["ELDER_WORD"][(tag, word, word_list[index-1], word_list[index-2])] = count + 1

                count = local_diff.setdefault("ELDER_WORD", {}).setdefault((pred_tag, word, word_list[index-1], word_list[index-2]), 0)
                local_diff["ELDER_WORD"][(pred_tag, word, word_list[index-1], word_list[index-2])] = count - 1

            #hypen feature
            if "-" in word:
                count = local_diff.setdefault("HYPEN", {}).setdefault(tag, 0)
                local_diff["HYPEN"][tag] = count + 1
                count = local_diff.setdefault("HYPEN", {}).setdefault(pred_tag, 0)
                local_diff["HYPEN"][pred_tag] = count - 1


            prev_tag        = tag
            pred_prev_tag   = pred_tag
            prev_word       = word


        for tag_type, info_hash in local_diff.iteritems():
            for key, value in info_hash.iteritems():
                count = self.weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.weights[tag_type][key] = count + value
                wt = self.avg_weights.setdefault(tag_type, {}).setdefault(key, 0)
                self.avg_weights[tag_type][key] = wt + multiplier * value


