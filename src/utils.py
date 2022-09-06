"""### Description
Generic utility functions.

### Functions
data_load
    Extract columns from file.
data_save
    Write columns to file. Overwrite if file exists.
data_append
    Append columns to file.
"""


import os
import csv
import glob
import typing


cast_hash = {
    "float": float,
    "str": str,
    "int": int
}


def data_load(
    path: str, *indices: int, firstrow: int = 0, lastrow: typing.Optional[int] = None,
    cast: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None
) -> typing.Tuple[list, ...]:
    """### Description
    Extract columns from file.

    ### Parameters
    path : str
        Path to file.
    indices : int
        Indices of columns to extract.
    firstrow : int, optional
        First data row number. Use this to skip headers.
    cast: str | Sequence[str], optional
        Specify data type for the extracted set or 
        for each column separately.
            "float" : floating point, default,
            "str" : string
            "int" : integer

    ### Returns
    data : tuple(list, ..)
        Tuple containing the separated data columns.
    """

    print("Loading", path)

    if not cast:
        cast = ["float"]*len(indices)
    elif isinstance(cast, str):
        cast = [cast]*len(indices)
    else:
        cast = list(cast)

    if len(cast) < len(indices):
        raise RuntimeError("data_load, unexpected number of arguments!")

    payload = [[] for _ in indices]
    max_index = max(indices)

    if os.path.exists(path):

        with open(path, 'r') as data:
            data_parsed = []
            if lastrow:
                data_parsed= data.readlines()[firstrow:lastrow+1]
            else:
                data_parsed= data.readlines()[firstrow:]
            for row in data_parsed:
                raw = row.split()
                if len(raw)>max_index:
                    for i, index in enumerate(indices):
                        payload[i].append(cast_hash[cast[i]](raw[index]))
    
    return tuple(payload)


def data_save(path: str, *columns: typing.Sequence) -> None:
    """### Description
    Write columns to file. Overwrite if file exists.

    ### Parameters
    path : str
        Path to file.
    columns : int
        Data columns
    """

    with open(path, 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ' ')
        writer.writerows([[*els] for els in zip(*columns)])


def data_append(path: str, *columns: typing.Sequence) -> None:
    """### Description
    Append columns to file.

    ### Parameters
    path : str
        Path to file.
    columns : int
        Data columns
    """

    with open(path, 'a', newline = '') as file:
        writer = csv.writer(file, delimiter = ' ')
        writer.writerows([[*els] for els in zip(*columns)])


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

class _cached:

    def __init__(self, func, defs: typing.Optional[dict] = None):
        
        print("Initializing cach!")

        self.func = func

        defs = {} if not defs else defs

        cached_defs = {}
        temp = data_load(
            ".cache/defs.dat", 0, 1,
            cast = ["str", "float"]
        )
        for key, val in data_load(
            ".cache/defs.dat", 0, 1,
            cast = ["str", "float"]
        ):
            cached_defs[key] = val

        if not defs == cached_defs:

            for file in glob.glob('.cache/*'): os.remove(file)

            data_save(
                ".cache/defs.dat",
                tuple(defs.keys()),
                tuple(defs.values())
            )        

    def __call__(self, *args, **kwds):
        print("Calling cach!")
        return self.func(*args, **kwds)


def cached(func: typing.Optional[typing.Callable] = None, defs: typing.Optional[dict] = None):
    if func:
        return _cached(func, defs)
    else:
        def wrapper(func):
            return _cached(func, defs)
        return wrapper