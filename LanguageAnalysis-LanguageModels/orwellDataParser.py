class OrwellDataParser:
    def __init__(self, logger, stop_token="<S>"):
        self.logger     = logger
        self.stop_token  = stop_token

    def iterate_line(self, orwell_file):
        self.logger.info("Started Iterating over Orwell data file %s" % orwell_file)
        counter = 0
        file_ptr    = open(orwell_file, "r")
        line = file_ptr.readline()
        while line:
            counter  += 1
            data_list = line.strip().split(' ')
            data_list.append(self.stop_token)
            yield data_list
            line = file_ptr.readline()
        self.logger.info("Completed parsing %s lines in %s" % (counter, orwell_file))
        file_ptr.close()


    
        
            
            
        
