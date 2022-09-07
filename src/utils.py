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
import shelve
import typing
import sqlite3
import functools
import collections
import _thread


cast_hash = {
    "float": float,
    "str": str,
    "int": int
}


def data_load(
    path: str, *indices: int, firstrow: int = 0, lastrow: typing.Optional[int] = None,
    cast: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    verbose: bool = False
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

    if verbose:
        print("Loading", path)

    if not cast:
        cast = ["float" for _ in indices]
    elif isinstance(cast, str):
        cast = [cast for _ in indices]
    else:
        cast = list(cast)

    if len(cast) < len(indices):
        raise RuntimeError("data_load, unexpected number of arguments!")

    payload = [list() for _ in indices]
    max_index = max(indices)

    if os.path.exists(path):

        with open(path, 'r') as data:
            data_parsed = []
            if lastrow:
                data_parsed= data.readlines()[firstrow:lastrow+1]
            else:
                data_parsed= data.readlines()[firstrow:]
            for row in data_parsed:
                raw = row.split(';')
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
        writer = csv.writer(file, delimiter = ';')
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
        writer = csv.writer(file, delimiter = ';')
        writer.writerows([[*els] for els in zip(*columns)])


def simple_cache(func):
    func.cache = {}
    #@functools.wraps(func)
    def wrapper(*args, **kwds):
        key = args
        result = func.cache.get(key)
        if result:
            return result
        func.cache[key] = result = func(*args, **kwds)
        return result
    return wrapper


#implement this as function decorator!
#https://www.datacamp.com/tutorial/decorators-python
#https://stackoverflow.com/questions/2392017/sqlite-or-flat-text-file
#use sqlite for cached data storage
#class cache_nokwargs:
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
#
#class _cached:
#
#    def __init__(self, func, defs: typing.Optional[dict] = None):
#
#        self.func = func
#        self.filename = ".cache/"+self.func.__module__+'.'+self.func.__name__
#
#        defs = {} if not defs else defs
#
#        cached_defs = {}
#        for key, val in zip(
#            *data_load(self.filename+".defs", 0, 1, cast = ["str", "float"])
#        ):
#            cached_defs[key] = val
#
#        if  (not defs == cached_defs) or \
#            (not os.path.exists(self.filename+".defs")) or \
#            (not os.path.exists(self.filename+".db")):
#
#            for file in glob.glob(self.filename+".*"):
#                os.remove(file)
#
#            data_save(
#                self.filename+".defs",
#                tuple(defs.keys()),
#                tuple(defs.values())
#            )
#
#            self.db_con = sqlite3.connect(self.filename+".db")
#            self.db_cur = self.db_con.cursor()
#
#            self.db_cur.executescript("""
#                DROP TABLE IF EXISTS cache;
#                CREATE TABLE IF NOT EXISTS cache (
#                    record_id INTEGER PRIMARY KEY,
#                    args VARCHAR(255),
#                    kwds VARCHAR(255),
#                    output DOUBLE
#            )""")
#
#        else:
#            self.db_con = sqlite3.connect(self.filename+".db")
#            self.db_cur = self.db_con.cursor()
#
#    def __call__(self, *args, **kwds):
#        self.db_cur.execute("SELECT output FROM cache WHERE args=? AND kwds=?", (str(args), str(kwds)))
#        cached_data = self.db_cur.fetchall()
#        if len(cached_data)>0:
#            return float(cached_data[0][0])
#        else:
#            output = self.func(*args, **kwds)
#            self.db_cur.execute("INSERT INTO cache(args, kwds, output) VALUES (?, ?, ?)", (str(args), str(kwds), output))
#            return output
#
#    def __del__(self):
#        self.db_con.commit()
#        self.db_con.close()
#
#
#def cached(func: typing.Optional[typing.Callable] = None, defs: typing.Optional[dict] = None):
#    """### Description
#    Decorator for caching function output. If the function was previously 
#    called with the same args and kwds, the result will be read from memory
#    instead of performing a repeat calculation. The cache will be reset on
#    change of global parameters.
#
#    ### Parameters
#    func: callable
#        Decorated function
#    defs: dict, optional
#        Dictionary of global parameters.
#
#    ### Returns
#    decorated_func:
#        Output of the decorated function.
#
#    ### Examples
#
#    Function accessing global parameters:
#    >>> @utils.cached(defs = pnjl.defaults.get_all_defaults(split_dict=True))
#        def Tc(mu: float) -> float:
#            TC0 = pnjl.defaults.TC0
#            KAPPA = pnjl.defaults.KAPPA
#            return math.fsum([TC0, -TC0*KAPPA*((mu/TC0)**2)])
#
#    Function independent of any global parameters (check if the function doesn't 
#    internaly call another function that does depend on globals!)
#    >>> @utils.cached
#        def En(p: float, mass: float) -> float:
#            body = math.fsum([p**2, mass**2])
#            return math.sqrt(body)
#    """
#
#    if func:
#        return _cached(func, defs)
#    else:
#        def wrapper(func):
#            return _cached(func, defs)
#        return wrapper
#
#
#class _memcached:
#
#    def __init__(self, func, defs: typing.Optional[dict] = None):
#
#        self.func = func
#        self.filename = ".memcache/"+self.func.__module__+'.'+self.func.__name__
#        self.mem = {}
#
#        defs = {} if not defs else defs
#
#        cached_defs = {}
#        for key, val in zip(
#            *data_load(self.filename+".defs", 0, 1, cast = ["str", "float"])
#        ):
#            cached_defs[key] = val
#
#        for key, val in zip(
#            *data_load(self.filename+".db", 0, 1, cast = ["str", "float"])
#        ):
#            self.mem[key] = val
#
#        if  (not defs == cached_defs) or \
#            (not os.path.exists(self.filename+".defs")) or \
#            (not os.path.exists(self.filename+".db")):
#
#            for file in glob.glob(self.filename+".*"):
#                os.remove(file)
#
#            data_save(
#                self.filename+".defs",
#                tuple(defs.keys()),
#                tuple(defs.values())
#            )
#
#    def __call__(self, *args, **kwds):
#        input = str(tuple([str(args), str(kwds)]))
#        if input not in self.mem:
#            output = self.func(*args, **kwds)
#            data_append(self.filename+".db", [input], [output])
#            self.mem[input] = output
#        return self.mem[input]
#
#
#def memcached(func: typing.Optional[typing.Callable] = None, defs: typing.Optional[dict] = None):
#    if func:
#        return _memcached(func, defs)
#    else:
#        def wrapper(func):
#            return _memcached(func, defs)
#        return wrapper