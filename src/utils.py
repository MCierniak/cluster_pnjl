"""Generic utility functions

### Functions
data_collect
    Extract columns from file.
"""


import typing


def data_collect(path: str, *indices: int, firstrow: int = 0) -> typing.Tuple[list, ...]:
    """Extract columns from file.

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
        for row in data.readlines()[firstrow:]:
            raw = row.split()
            if len(raw)>max_index:
                for i, index in enumerate(indices):
                    payload[i].append(raw[index])
    
    return tuple(payload)

