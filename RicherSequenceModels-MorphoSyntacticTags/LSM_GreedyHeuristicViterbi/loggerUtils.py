#
#Author		: Kumaresh Visakan Murugan
#Email		: mvisakan@cs.wisc.edu
#Class		: CS545 - NLP
#Instructor	: Prof. Benjamin Snyder
#

import logging

def initialize_logger(filename="default.log"):
	logger = logging.getLogger('')
	handler = logging.FileHandler(filename)
	formatter = logging.Formatter('%(asctime)s : <%(pathname)s:%(lineno)d> : %(levelname)s : %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)
	return logger

