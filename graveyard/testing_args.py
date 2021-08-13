import pickle
import numpy as np
import pandas as pd
import ssms
import argparse

def none_or_str(value):
    print('none_or_str')
    print(value)
    print(type(value))
    if value == 'None':
        return None
    return value

def none_or_int(value):
    print('none_or_int')
    print(value)
    print(type(value))
    #print('printing type of value supplied to non_or_int ', type(int(value)))
    if value == 'None':
        return None
    return int(value)

if __name__ == "__main__":

    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--my_list",
                     type = list,
                     default = None)
#     CLI.add_argument('--config_dict_key',
#                      type = none_or_int,
#                      default = None)
    
    args = CLI.parse_args()
    print(args)
    print(args.my_list)