import numpy as np


def n_size_ndarray_creation(n, dtype=np.int):
    X = np.arange(n*n).reshape(n,n)
    return X

def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    
    if type == 0 : 
        result = np.zeros(shape, dtype = dtype)
    elif type == 1 :
        result = np.ones(shape, dtype = dtype)
    else : 
        result = np.empty(shape, dtype = dtype)
    
    return result


def change_shape_of_ndarray(X, n_row):
    if n_row == 1 : 
        result = X.reshape(n_row, -1).flatten()
    else : 
        result = X.reshape(n_row, -1)
        
    return result


def concat_ndarray(X_1, X_2, axis):
    try : 
        X_1.shape[1]
    except :
        X_1 = X_1.reshape(1, -1)
        
    try :    
        X_2.shape[1]
    
    except :
        
        X_2 = X_2.reshape(1, -1)
        
    try :
        
        result = np.concatenate((X_1, X_2), axis = axis)
    
    except Exception :
        result = False
    
    
    
    return result


def normalize_ndarray(X, axis=99, dtype=np.float32):
    if axis == 0 :
        Z = (X - X.mean(axis = axis)) / X.std(axis = axis)
    elif axis == 1 :
        n_row = X.shape[0]
        Z = (X - X.mean(axis = axis).reshape(n_row, -1)) / X.std(axis = axis).reshape(n_row, -1)
    else : 
        Z = (X - X.mean()) / X.std()
    
    
    return Z


def save_ndarray(X, filename="test.npy"):
    return np.save(filename, arr = X)


def boolean_index(X, condition):
    return X[eval("X" + condition)]


def find_nearest_value(X, target_value):
    absolute_value = np.abs(X - target_value)
    min_value = np.argmin(absolute_value)
    
    return X[min_value]


def get_n_largest_values(X, n):
    ls = []
    target_value = np.max(X)
    ls.append(target_value)
    for _ in range(n-1) : 
        X = X[X < target_value]
        target_value = np.max(X)
        ls.append(target_value)
        
        
    return np.array(ls)
