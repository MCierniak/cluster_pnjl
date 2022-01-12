import numpy as np

def data_collect(i,j,path):
    print("Loading", path)
    x = []
    y = []
    with open(path) as data:
        for row in data.readlines()[0:]:
            x.append(row.split()[i])
            y.append(row.split()[j])
        x = np.array(x).astype(float)
        y = np.array(y).astype(float)
    
    return (x, y)