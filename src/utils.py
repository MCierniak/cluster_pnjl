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
import sty
import glob
import typing
import pickle
import atexit
import hashlib
import functools


EXP_LIMIT = 709.78271
CACHE_FOLDER = ".cache/"
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


def flush_cache():
    """### Description
    Remove all .cache files.
    """
    
    for file in glob.glob(CACHE_FOLDER+'*.cache'):
            os.remove(file)


class cached:
    """### Description
    Persistent cache decorator. Warning! Instances will not get garbage 
    collected, use only as toplevel function decorator!
    """

    def __init__(self, func):
        self.func = func
        self.filepath = CACHE_FOLDER \
                        +self.func.__module__ \
                        +'.' \
                        +self.func.__name__ \
                        +'.cache'
        try:
            with open(self.filepath, "rb") as file:
                self.cache = pickle.load(file)
        except FileNotFoundError as e:
            self.cache = {}

        atexit.register(self.cleanup)

    @functools.lru_cache
    def __call__(self, *args, **kwds):
        key = ""
        if args:
            for arg in args:
                if not callable(arg):
                    key += str(arg) + ", "
        key += str(kwds)
        try:
            return self.cache[key]
        except KeyError:
            self.cache[key] = result = self.func(*args, **kwds)
            return result

    def cleanup(self):
        with open(self.filepath, "wb") as file:
            pickle.dump(self.cache, file)


def md5(fname: str) -> str:
    """### Description
    Calculate the md5 checksum of a file.

    ### Parameters
    fname : str
        Path to file.

    ### Returns
    md5 : str
        Md5 checksum.
    """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_checksum(file_path: str):
    """### Description
    Verify checksum of a file againts an existing pickled value. 
    Flush cache if failed.
    """

    file_base_name = os.path.relpath(file_path).replace("\\", ".")
    hash_savefile_path = CACHE_FOLDER+file_base_name+".checksum"

    current_hash = md5(file_path)
    
    if os.path.exists(hash_savefile_path):

        with open(hash_savefile_path, "rb") as file:
            saved_hash = pickle.load(file)

        if saved_hash != current_hash:

            print(
                sty.bg.red+file_base_name,
                "changed, flushing cache.",
                sty.rs.all
            )

            flush_cache()

            with open(hash_savefile_path, "wb") as file:
                pickle.dump(current_hash, file) 

        else:
            print(
                sty.bg.green + sty.fg.black + file_base_name,
                "unchanged." + sty.rs.all
            )

    else:

        print(
            sty.bg.red+file_base_name,
            "changed, flushing cache.",
            sty.rs.all
        )

        flush_cache()

        with open(hash_savefile_path, "wb") as file:
            pickle.dump(current_hash, file)