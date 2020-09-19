import optparse
import logging
import re


def model_choice_dataset_args(usage):
    parser = optparse.OptionParser(usage)
    parser.add_option('-m', dest='m', help='Model', type='str', default='model')
    parser.add_option('-d', dest='d', help='Whole Dataset', type='str', default='foursq2014_TKY_node_format.txt')

    options, args = parser.parse_args()

    return options, args


def log_def(log_file_name="log.log"):
    logging.basicConfig(filename=log_file_name, filemode="w", format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                        level=logging.INFO)


def parse_model_name(model_name):
    if re.match(model_name, '.emb$'):
        return 'emb_file'
    else:
        return 'model_file'

