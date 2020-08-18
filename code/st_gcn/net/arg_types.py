import configargparse as argparse
import re
from collections import OrderedDict


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_tuple(cast_type):
    regex = re.compile(r'\d+\.\d+|\d+')

    def parse_tuple(v):
        vals = regex.findall(v)
        return [cast_type(val) for val in vals]

    return parse_tuple


def arg_list_tuple(cast_type):
    regex = re.compile(r'\([^\)]*\)')
    tuple_parser = arg_tuple(cast_type)

    def parse_list(v):
        tuples = regex.findall(v)
        return [tuple_parser(t) for t in tuples]

    return parse_list


def arg_dict(cast_type):
    regex_pairs = re.compile(r'[^\ ]+=[^\ ]+')
    regex_keyvals = re.compile(r'([^\ ]+)=([^\ ]+)')

    def parse_dict(v):
        d = OrderedDict()
        for keyval in regex_pairs.findall(v):
            key, val = regex_keyvals.match(keyval).groups()
            d.update({key:cast_type(val)})
        return d

    return parse_dict
