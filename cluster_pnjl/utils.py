import numpy as np

def data_collect(i, j, path, firstrow = 0):
    print("Loading", path)
    x = []
    y = []
    with open(path) as data:
        for row in data.readlines()[firstrow:]:
            if not row == '\n':
                x.append(row.split()[i])
                y.append(row.split()[j])
        x = np.array(x).astype(float)
        y = np.array(y).astype(float)
    
    return (x, y)