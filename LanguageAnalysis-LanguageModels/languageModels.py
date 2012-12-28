import os
import math
import gmpy2
import random

class GeometricModel:
    def __init__(self, logger, data_parser, p_val, q_val):
        self.logger = logger
        self.data_parser = data_parser
        self.p_val  = p_val
        self.q_val  = q_val

    def compute_prob_plots(self, sent_start_len=1, sent_end_len=100):
        x_list = []
        y_list = []
        index = sent_start_len
        while index <= sent_end_len:
            x_list.append(index)
            y_list.append(pow(self.p_val, index) * self.q_val)
            index += 1
        return x_list, y_list

    def compute_freq_plots(self, orwell_file):
        sent_meta = {}
        for token_list in self.data_parser.iterate_line(orwell_file):
            count = sent_meta.setdefault(len(token_list) - 1, 0)
            sent_meta[len(token_list) - 1] = count + 1

        sent_list = sorted(sent_meta.iteritems(), key=lambda x:x[0])
        x_list  = [x for x, y in sent_list]
        y_list  = [y for x, y in sent_list]
        return x_list, y_list

    def compute_parameters(self, orwell_file):
        data_prob = 0.0
        total_count = 0
        for token_list in self.data_parser.iterate_line(orwell_file):
            total_count += 1
            l = len(token_list) - 1
            sent_prob = pow(self.p_val, l) * self.q_val
            data_prob += gmpy2.log2(sent_prob)
        log_data        = (data_prob / total_count)
        perplexity      = math.pow(2, (-1.0 * log_data))
        cross_entropy   = math.log(perplexity, 2)
        return cross_entropy, perplexity
        

class MultinomialModel:
    def __init__(self, logger, data_parser):
        self.logger = logger
        self.data_parser = data_parser
        self.sent_meta = {}
        self.sent_count = 0

    def train(self, orwell_file):
        for token_list in self.data_parser.iterate_line(orwell_file):
            sent_len = len(token_list) - 1
            self.sent_count += 1
            count = self.sent_meta.setdefault(sent_len, 0)
            self.sent_meta[sent_len] = count + 1

    def sent_prob(self, sent_len):
        return (self.sent_meta.get(sent_len, 0) * 1.0) / (self.sent_count)

    def compute_prob_plots(self, sent_start_len=1, sent_end_len=100):
        x_list = []
        y_list = []
        index = sent_start_len
        while index <= sent_end_len:
            x_list.append(index)
            y_list.append(self.sent_prob(index))
            index += 1
        return x_list, y_list

    def compute_parameters(self, orwell_file):
        data_prob = 0.0
        total_count = 0
        for token_list in self.data_parser.iterate_line(orwell_file):
            total_count += 1
            sent_len = len(token_list) - 1
            sent_prob = self.sent_prob(sent_len)
            data_prob += gmpy2.log2(sent_prob)
        log_data        = (data_prob / total_count)
        perplexity      = math.pow(2, (-1.0 * log_data))
        cross_entropy   = math.log(perplexity, 2)
        return cross_entropy, perplexity
        

class NegativeBinomialModel:
    def __init__(self, logger, data_parser, p_val, r_val):
        self.logger = logger
        self.data_parser = data_parser
        self.p_val = p_val
        self.r_val = r_val

    def sent_prob(self, sent_len):
        factor1 = (math.gamma(self.r_val + sent_len)) / (math.gamma(self.r_val) * math.gamma(sent_len + 1))
        factor2 = pow(self.p_val, sent_len) * pow((1.0 - self.p_val), self.r_val)
        return factor1 * factor2

    def compute_prob_plots(self, sent_start_len=1, sent_end_len=100):
        x_list = []
        y_list = []
        index  = sent_start_len
        while index <= sent_end_len:
            x_list.append(index)
            y_list.append(self.sent_prob(index))
            index += 1
        return x_list, y_list

    def compute_parameters(self, orwell_file):
        data_prob = 0.0
        total_count = 0
        for token_list in self.data_parser.iterate_line(orwell_file):
            total_count += 1
            sent_len = len(token_list) - 1
            sent_prob = self.sent_prob(sent_len)
            data_prob += gmpy2.log2(sent_prob)
        log_data        = (data_prob / total_count)
        perplexity      = math.pow(2, (-1.0 * log_data))
        cross_entropy   = math.log(perplexity, 2)
        return cross_entropy, perplexity
        

class sentenceGenerator:
    def __init__(self, logger, obj_model, stop_token="<S>"):
        self.logger         = logger
        self.obj_model      = obj_model
        self.stop_token     = stop_token

    def generate(self):
        if isinstance(self.obj_model, Unigram):
            return self.generate_unigram()
        elif isinstance(self.obj_model, Bigram):
            return self.generate_bigram()
        elif isinstance(self.obj_model, Trigram):
            return self.generate_trigram()
        elif isinstance(self.obj_model, InterpolatedModel):
            return self.generate_interpolated()

    def generate_interpolated(self):
        words_list = []
        elder_word = None
        prev_word  = None
        while True:
            rand_no = random.uniform(0, 1)
            l_obj = self.obj_model.generate_obj_model(rand_no)
            rand_no = random.uniform(0, 1)
            if isinstance(l_obj, Unigram):
                word = l_obj.generate_word(rand_no)
            elif isinstance(l_obj, Bigram):
                if prev_word is None:
                    word = l_obj.generate_first_word(rand_no)
                else:
                    word = l_obj.generate_word(rand_no, prev_word)
            elif isinstance(l_obj, Trigram):
                if elder_word is None:
                    if prev_word is None:
                        word = l_obj.generate_first_word(rand_no)
                    else:
                        word = l_obj.generate_second_word(rand_no, prev_word)
                else:
                    word = l_obj.generate_word(rand_no, prev_word, elder_word)
            if word is None:
                continue
            words_list.append(word)
            elder_word = prev_word
            prev_word = word
            if word == self.stop_token:
                break
        return " ".join(words_list)
        

    def generate_trigram(self):
        words_list = []

        rand_no = random.uniform(0, 1)
        word = self.obj_model.generate_first_word(rand_no)
        words_list.append(word)
        elder_word = word

        rand_no = random.uniform(0, 1)
        word = self.obj_model.generate_second_word(rand_no, elder_word)
        words_list.append(word)
        prev_word = word
        
        while True:
            rand = random.uniform(0, 1)
            word = self.obj_model.generate_word(rand_no, prev_word, elder_word)
            words_list.append(word)
            if word == self.stop_token:
                break
            elder_word = prev_word
            prev_word = word

        return " ".join(words_list)

    def generate_bigram(self):
        words_list = []
        rand_no = random.uniform(0, 1)
        word = self.obj_model.generate_first_word(rand_no)
        words_list.append(word)
        prev_word = word
        while True:
            rand_no = random.uniform(0, 1)
            word = self.obj_model.generate_word(rand_no, prev_word)
            if word is None:
                break
            words_list.append(word)
            if word == self.stop_token:
                break
            prev_word = word
        return " ".join(words_list)

    def generate_unigram(self):
        words_list = []
        while True:
            rand_no = random.uniform(0, 1)
            word = self.obj_model.generate_word(rand_no)
            words_list.append(word)
            if word == self.stop_token:
                break
        return " ".join(words_list)
            
def compute_cross_entropy_perplexity(obj_model, data_file):
    obj_model.logger.info("Started analysing data file %s to compute cross entropy" % data_file)
    data_prob  = 0.0
    word_count = 0
    for token_list in obj_model.data_parser.iterate_line(data_file):
        word_count += len(token_list)
        sent_prob   = obj_model.sentence_probability(token_list)
        if sent_prob is None:
            return None
        data_prob  += gmpy2.log2(sent_prob)
    obj_model.logger.info("Data Probability : %.10f" % data_prob)
    log_data        = (data_prob / word_count)
    obj_model.logger.info("Log Data         : %.10f" % log_data)
    perplexity      = math.pow(2, (-1.0 * log_data))
    obj_model.logger.info("Perplexity       : %.10f" % perplexity)
    cross_entropy   = math.log(perplexity, 2)
    obj_model.logger.info("Cross Entropy    : %.10f" % cross_entropy)
    return cross_entropy, perplexity

class Unigram:
    def __init__(self, logger, data_parser, vocab_count, stop_token="<S>", smoothing_factor=0.00001):
        self.logger             = logger
        self.data_parser        = data_parser
        self.stop_token         = stop_token
        self.vocab_count        = vocab_count
        self.s_factor           = smoothing_factor
        self.word_freq_dict     = {}
        self.tr_word_count      = 0
        self.base_value         = 0.0
        self.word_list          = []

    def train(self, train_data_file):
        #get freq of words in the train data
        self.logger.info("Started training data for unigram model")
        for token_list in self.data_parser.iterate_line(train_data_file):
            for word_token in token_list:
                self.tr_word_count             += 1
                word_freq                       = self.word_freq_dict.setdefault(word_token, 0)
                self.word_freq_dict[word_token] = word_freq + 1
        self.logger.info("Completed analyzing training data")
        self.logger.info("Distinct word types : %d" % len(self.word_freq_dict))
        self.logger.info("Word count          : %d" % self.tr_word_count)
        self.base_value = self.tr_word_count + (self.vocab_count * self.s_factor)
        self.logger.info("Base value          : %.10f" % self.base_value)

        for word in self.word_freq_dict:
            word_prob = self.word_probability(word)
            self.word_list.append((word, word_prob))

    def get_non_stop_probability(self):
        prob_sum = 0.0
        for word, prob in self.word_list:
            if word == self.stop_token:
                continue
            prob_sum += prob
        return prob_sum
            


    def word_probability(self, word_token):
        if self.base_value == 0:
            self.logger.info("Base value not computed!!! Please train data")
            return None
        word_freq   = self.word_freq_dict.get(word_token, 0)
        return (word_freq + self.s_factor) / (self.base_value)

    def sentence_probability(self, token_list):
        if not token_list:
            return None
        sent_prob   = 1.0
        for token in token_list:
            word_prob = self.word_probability(token)
            if word_prob is None:
                return None
            sent_prob = gmpy2.mpfr(sent_prob * word_prob)
        return sent_prob

    def generate_word(self, rand_no):
        cum_prob = 0.0
        for word, word_prob in self.word_list:
            cum_prob += word_prob
            if rand_no < cum_prob:
                return word
        self.logger.info("Unknown Rand !!!")
        return None

    def test(self):
        cum_prob = 0.0
        for word, word_prob in self.word_list:
            if word == self.stop_token:
                continue
            cum_prob += word_prob
        print "TEST: REST PROB : %.10f" % cum_prob
        print "TEST: STOP PROB : %.10f" % self.word_probability(self.stop_token)
        

class Bigram:
    def __init__(self, logger, data_parser, vocab_count, stop_token="<S>", s_factor=0.00001):
        self.logger         = logger
        self.data_parser    = data_parser
        self.stop_token     = stop_token
        self.vocab_count    = vocab_count
        self.s_factor       = s_factor
        self.bigrams        = {}
        self.word_freq      = {}
        self.start_words    = {}
        self.tr_word_count  = 0
        self.tr_sent_count  = 0


    def train(self, train_data_file):
        self.logger.info("Started training data for Bigram model")
        for token_list in self.data_parser.iterate_line(train_data_file):
            self.tr_sent_count += 1
            for index, token in enumerate(token_list):
                self.tr_word_count  += 1  

                count = self.word_freq.setdefault(token, 0)
                self.word_freq[token] = count + 1

                if index > 0:
                    prev_word = token_list[index - 1]
                    bigram_dict = self.bigrams.setdefault(prev_word, {})
                    count = bigram_dict.setdefault(token, 0)
                    bigram_dict[token] = count + 1
                else:
                    count = self.start_words.setdefault(token, 0)
                    self.start_words[token] = count + 1

        self.logger.info("Completed analyzing data")
        self.logger.info("Distinct bigrams      : %d" % len(self.bigrams))
        self.logger.info("Distinct word tokens  : %d" % len(self.word_freq))
        self.logger.info("Word count            : %d" % self.tr_word_count)



    def word_probability(self, word_token, prev_word=None):
        if not word_token:
            self.logger.info("word_token not found !!!")
            return None

        if prev_word is None:
            word_freq = self.start_words.get(word_token, 0)
            return (word_freq + self.s_factor) / (self.tr_sent_count + (self.vocab_count * self.s_factor))

        word_freq = self.bigrams.get(prev_word, {}).get(word_token, 0)
        base_value = (self.vocab_count * self.s_factor) + self.word_freq.get(prev_word, 0)
        return (word_freq + self.s_factor) / (base_value)


    def sentence_probability(self, token_list):
        if not token_list:
            return None
        sent_prob = 1.0
        for index, token in enumerate(token_list):
            if index == 0:
                word_prob = self.word_probability(token)
            else:
                word_prob  = self.word_probability(token, token_list[index - 1])
            if word_prob is None:
                return None
            sent_prob = gmpy2.mpfr(sent_prob * word_prob)
        return sent_prob

    def generate_first_word(self, rand_no):
        cum_prob = 0.0
        start_word_list = [(word, self.word_probability(word)) for word in self.start_words]
        for word, prob in start_word_list:
            cum_prob += prob
            if rand_no < cum_prob:
                return word

    def generate_word(self, rand_no, prev_word):
        cum_prob = 0.0
        word_dict = self.bigrams.get(prev_word, {})
        if not word_dict:
            return None
        word_list = [(word, self.word_probability(word, prev_word)) for word in word_dict]
        for word, prob in word_list:
            cum_prob += prob
            if rand_no < cum_prob:
                return word
        return None
        
            
                    

class Trigram:
    def __init__(self, logger, data_parser, vocab_count, stop_token="<S>", s_factor=0.00001):
        self.logger             = logger
        self.data_parser        = data_parser
        self.stop_token         = stop_token
        self.vocab_count        = vocab_count
        self.s_factor           = s_factor
        self.trigrams           = {}
        self.bigrams            = {}
        self.start_words        = {}
        self.start_tuples       = {}
        self.tr_sent_count      = 0


    def train(self, train_data_file):
        self.logger.info("Started training data for Trigram model")        
        for token_list in self.data_parser.iterate_line(train_data_file):
            self.tr_sent_count += 1
            for index, word_token in enumerate(token_list):
                if index == 0:
                    count = self.start_words.setdefault(word_token, 0)
                    self.start_words[word_token] = count + 1

                elif index == 1:
                    start_tuples_dict = self.start_tuples.setdefault(token_list[index - 1], {})
                    count = start_tuples_dict.setdefault(word_token, 0)
                    start_tuples_dict[word_token] = count + 1

                else:
                    t_dict = self.trigrams.setdefault((token_list[index - 2], token_list[index - 1]), {})
                    count  = t_dict.setdefault(word_token, 0)
                    t_dict[word_token] = count + 1
                    
                    count = self.bigrams.setdefault((token_list[index - 2], token_list[index - 1]), 0)
                    self.bigrams[(token_list[index - 2], token_list[index - 1])] = count + 1

        self.logger.info("Completed analyzing data")
        self.logger.info("Distinct trigrams         : %d" % len(self.trigrams))
        self.logger.info("Distinct bigrams          : %d" % len(self.bigrams))

    def word_probability(self, word_token, prev_word=None, elder_word=None):
        if not word_token:
            self.logger.info("No word token !!!")
            return None

        if elder_word is None:
            if prev_word is None:
                word_freq = self.start_words.get(word_token, 0)
                return (word_freq + self.s_factor) / (self.tr_sent_count + (self.vocab_count * self.s_factor))
            else:
                word_freq = self.start_tuples.get(prev_word, {}).get(word_token, 0)
                return (word_freq + self.s_factor) / (self.start_words.get(prev_word, 0) + (self.vocab_count * self.s_factor))

        word_freq = self.trigrams.get((elder_word, prev_word), {}).get(word_token, 0)
        return (word_freq + self.s_factor) / (self.bigrams.get((elder_word, prev_word), 0) + (self.vocab_count * self.s_factor))

    def sentence_probability(self, token_list):
        if not token_list:
            return None
        sent_prob = 1.0
        for index, token in enumerate(token_list):
            if index == 0:
                word_prob = self.word_probability(token)
            elif index == 1:
                word_prob = self.word_probability(token, token_list[index - 1])
            else:
                word_prob  = self.word_probability(token, token_list[index - 1], token_list[index - 2])
            if word_prob is None:
                return None
            sent_prob = gmpy2.mpfr(sent_prob * word_prob)
        return sent_prob

    def generate_first_word(self, rand_no):
        cum_prob = 0.0
        words_list = [(word, self.word_probability(word)) for word in self.start_words]
        for word, prob in words_list:
            cum_prob += prob
            if rand_no < cum_prob:
                return word

    def generate_second_word(self, rand_no, prev_word):
        cum_prob = 0.0
        words_dict = self.start_tuples.get(prev_word, {})
        words_list = [(word, self.word_probability(word, prev_word)) for word in words_dict]
        for word, prob in words_list:
            cum_prob += prob
            if rand_no < cum_prob:
                return word

    def generate_word(self, rand_no, prev_word, elder_word):
        cum_prob = 0.0
        words_dict = self.trigrams.get((elder_word, prev_word), {})
        words_list = [(word, self.word_probability(word, prev_word, elder_word)) for word in words_dict]
        for word, prob in words_list:
            cum_prob += prob
            if rand_no < cum_prob:
                return word
        

   
class InterpolatedModel:
    def __init__(self, logger, data_parser, u_obj, b_obj, t_obj, lambda1=(1.0/3.0), lambda2=(1.0/3.0), lambda3=(1.0/3.0)):
        self.logger             = logger
        self.data_parser        = data_parser
        self.u_obj              = u_obj
        self.b_obj              = b_obj
        self.t_obj              = t_obj
        self.lambda1            = lambda1
        self.lambda2            = lambda2
        self.lambda3            = lambda3

    def word_probability(self, word_token, prev_token=None, elder_token=None):
        u_prob   = self.u_obj.word_probability(word_token)
        b_prob   = self.b_obj.word_probability(word_token, prev_token)
        t_prob   = self.t_obj.word_probability(word_token, prev_token, elder_token)
        return ((self.lambda1 * u_prob) + (self.lambda2 * b_prob) + (self.lambda3 * t_prob))

    def sentence_probability(self, token_list):
        if not token_list:
            return None
        sent_prob = 1.0
        for index, token in enumerate(token_list):
            if index == 0:
                word_prob = self.word_probability(token)
            elif index == 1:
                word_prob = self.word_probability(token, token_list[index - 1])
            else:
                word_prob = self.word_probability(token, token_list[index - 1], token_list[index - 2])
            if word_prob is None:
                return None
            sent_prob = gmpy2.mpfr(sent_prob * word_prob)
        return sent_prob

    def generate_obj_model(self, rand_val):
        cum_prob = 0.0
        for l_val, obj_model in [(self.lambda1, self.u_obj), (self.lambda2, self.b_obj), (self.lambda3, self.t_obj)]:
            cum_prob += l_val
            if rand_val < cum_prob:
                return obj_model
