"""Generic utility functions

### Functions
data_collect
    Extract columns from file.
"""


import os
import csv
import glob
import typing


def data_collect(path: str, *indices: int, firstrow: int = 0, lastrow: typing.Optional[int] = None) -> typing.Tuple[list, ...]:
    """### Description
    Extract columns from file.

    ### Parameters
    path : str
        Path to file.
    indices : int
        Indices of columns to extract.
    firstrow : int, optional
        First data row number. Use this to skip headers.

    ### Returns
    data : tuple(list, ..)
        Tuple containing the separated data columns.
    """

    print("Loading", path)

    payload = [[]]*len(indices)
    max_index = max(indices)

    with open(path) as data:
        data_parsed = []
        if lastrow:
            data_parsed= data.readlines()[firstrow:lastrow+1]
        else:
            data_parsed= data.readlines()[firstrow:]
        for row in data_parsed:
            raw = row.split()
            if len(raw)>max_index:
                for i, index in enumerate(indices):
                    payload[i].append(raw[index])
    
    return tuple(payload)

#implement this as function decorator!
#https://www.datacamp.com/tutorial/decorators-python
#https://stackoverflow.com/questions/2392017/sqlite-or-flat-text-file
#use sqlite for cached data storage
#class Memorize(object):
#    def __init__(self, func):
#        self.func = func
#        self.eval_points = {}
#    def __call__(self, *args):
#        if args not in self.eval_points:
#            self.eval_points[args] = self.func(*args)
#        return self.eval_points[args]

#remake as class... finish this...
#def cached(defs: typing.Optional[dict] = None):
#
#    def decorator(func):
#
#        defs = {} if not defs else defs
#
#        cached_defs = {}
#        for key, val in data_collect(".cache/defs.dat", 0, 1):
#            cached_defs[key] = val
#
#        if not defs == cached_defs:
#
#            for file in glob.glob('.cache/*'): os.remove(file)
#            
#            with open(".cache/defs.dat", 'w', newline = '') as file:
#                writer = csv.writer(file, delimiter = '\t')
#                writer.writerows([[key, val] for key, val in defs])
#
#        def wrapper(*fargs, **fkwargs):
#            return func(*fargs, **fkwargs)
#
#        return wrapper
#
#    return decorator

class cached:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)