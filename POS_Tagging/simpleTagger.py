class SimpleTagger:
    def __init__(self, logger, data_parser, start_tag="START", stop_tag="STOP"):
        self.logger             = logger
        self.data_parser        = data_parser
        self.start_tag          = start_tag
        self.stop_tag           = stop_tag
        self.word_info          = {}
        self.tags_blacklist     = ["START", "STOP"]
        self.default_tag        = None


    def train(self, training_file, end_line):
        self.logger.info("Started training data from %s upto line %d" % (training_file, end_line))
        tags = {}
        for line_no, word_list in self.data_parser.next(training_file):

            if line_no > end_line:
                break

            for word, tag in word_list:

                if tag in self.tags_blacklist:
                    continue

                tag_info        = self.word_info.setdefault(word, {})
                tag_count       = tag_info.setdefault(tag, 0)
                tag_info[tag]   = tag_count + 1

                tag_count       = tags.setdefault(tag, 0)
                tags[tag]       = tag_count + 1

        self.default_tag = self.get_max_tag(tags)
        self.logger.info("Simple Tagger Model trained with lines upto %d from %s" % (end_line, training_file))
        self.logger.info("DEFAULT TAGS          : %s" % self.default_tag)

    def reset(self):
        self.logger.info("Resetting Model parameters...")
        self.word_info      = {}
        self.default_tag    = None

    def predict(self, word):
        tags = self.word_info.get(word)
        if not tags:
            return self.default_tag
        return self.get_max_tag(tags)

    def is_unseen(self, word):
        return ( self.word_info.get(word) == None )

    def get_max_tag(self, tags_dict):
        if tags_dict:
            return sorted(tags_dict.iteritems(), key=lambda x:x[1], reverse=True)[0][0]
        return None
