class OrwellDataParser:
    def __init__(self, logger, start_token="<S>", stop_token="</S>"):
        self.logger         = logger
        self.start_token    = start_token
        self.stop_token     = stop_token

    def next(self, orwell_file):
        self.logger.info("Started Iterating over orwell file : %s" % orwell_file)
        counter     = 0
        file_ptr    = open(orwell_file, "r")
        line        = file_ptr.readline()
        while line:
            counter     += 1
            data_list    = [(self.start_token, "START")]
            for entity in line.strip().split(' '):
                word, tag = entity.split('\\', 1)
                data_list.append((word, tag))
            data_list.append((self.stop_token, "STOP"))
            yield counter, data_list
            line = file_ptr.readline()
        file_ptr.close()
        self.logger.info("Parsed %s lines in %s file" % (counter, orwell_file))
