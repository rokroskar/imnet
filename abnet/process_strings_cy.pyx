from Levenshtein import distance
import numpy as np
from collections import defaultdict
import cython
from libc.stdlib cimport abs

@cython.boundscheck(False)
def get_degrees_cython(int [:] idxs, list strings, int min_ld, int max_ld):
    cdef int size = len(strings)
    cdef int idx, my_idx, i
    cdef int ld, my_length, s_length
    
    degrees = np.zeros((size, max_ld-min_ld + 2), dtype='int32')
    
    deg_dict = defaultdict(list)

    for i in range(len(idxs)):
        my_idx = idxs[i]
        my_string = strings[my_idx]
        my_length = len(my_string)
        my_degrees = []
        
        # generate connections
        for idx in range(my_idx): 
            s = strings[idx]
            ld = distance(my_string, s)

            # if it's a connection, bump the degree count for source and destination
            if ld <= max_ld: 
                my_degrees.append(ld)
                deg_dict[idx].append(ld)

        deg_dict[my_idx] += my_degrees
        
        # save the degree count for my_idx 
        if i%100==0: 
            for idx in deg_dict.keys():
                degree_count = np.bincount(deg_dict[idx])
                degrees[idx, 0:len(degree_count)] += degree_count
            
            # reset the dictionary
            deg_dict = defaultdict(list)
    
    # clear the remaining items from temporary dictionary
    for idx in deg_dict.keys():
            degree_count = np.bincount(deg_dict[idx])
            degrees[idx, 0:len(degree_count)] += degree_count
        
    return degrees

@cython.boundscheck(False)
def generate_matrix_elements_cython(int [:] idxs, list strings, int min_ld, int max_ld): 
    cdef int nstrings = len(strings)
    cdef int idx, my_idx, i
    cdef int ld, my_length, s_length

    for i in range(len(idxs)):
        my_idx = idxs[i]
        my_string = strings[my_idx]

        # generate connections
        for idx in range(my_idx):
            s = strings[idx]
            ld = distance(my_string,s)

            # if it's a connection, yield the coordinates and the Levenshtein distance
            if ld <= max_ld: 
                yield (my_idx, idx, ld)
