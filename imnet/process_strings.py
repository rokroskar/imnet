#!/usr/bin/env python
from __future__ import absolute_import

import Levenshtein
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import os, sys
from warnings import warn
import csv
import itertools
import logging
import pandas as pd
from random import shuffle

from .process_strings_cy import get_degrees_cython, generate_matrix_elements_cython

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_matrix_elements_idx(iterator, strings, min_ld=1, max_ld=1):
    """
    Generates non-zero matrix elements by calculating the Levenshtein
    distance for each string in the partition against all the strings in the 
    lookup dictionary. It only searches the strings with 0 < idx < my_idx, 
    i.e. only filling in the upper triangle of the matrix.
    """
    try:
        import findspark

        findspark.init()
        from pyspark.sql import Row
    except:
        warn("Problem importing pyspark -- are you sure your SPARK_HOME is set?")

    for my_idx in iterator:
        for idx in range(my_idx):
            ld = Levenshtein.distance(strings[my_idx], strings[idx])
            if ld >= min_ld and ld <= max_ld:
                yield Row(src=my_idx, dst=idx, weight=int(ld))


def _balance_partitions(x, N_partitions, N_elements):
    """Balances out the number of elements across the partitions"""
    n_per_part = N_elements / N_partitions / 2

    part = 0
    if x > N_elements / 2:
        x = N_elements - x
    part = x / n_per_part
    return part if part < N_partitions else part - 1


def distance_matrix(strings, min_ld=1, max_ld=1, sc=None):
    """
    Given the set of strings, return a distance matrix
    
    Inputs
    ------

    strings : list 
        a list of strings to use for the pairwise distance matrix
    min_ld : int, optional
        minimum Levenshtein distance
    max_ld : int, optional
        maximum Levenshtein distance
    sc : pyspark.SparkContext
        a live SparkContext; if none is given, the calculation is done locally

    Returns
    -------

    If `sc` is specified, returns an RDD of (src, dst, distance) tuples. Otherwise a generator 
    of (src, dst, distance) tuples is returned that can be used to construct a sparse matrix or a graph. 
    """

    nstrings = len(strings)
    logger.info("number of strings " + str(nstrings))

    if sc is not None:
        # broadcast the lookup dictionaries
        strings_b = sc.broadcast(strings)

        # create an RDD of indices and balance partitioning
        number_of_partitions = sc.defaultParallelism * 10

        idxs = range(nstrings)
        shuffle(idxs)

        idx_rdd = sc.parallelize(idxs, number_of_partitions)

        mat = idx_rdd.mapPartitions(
            lambda x: generate_matrix_elements_cython(
                np.array(list(x), dtype=np.int32), strings_b.value, min_ld, max_ld
            )
        )

    else:
        mat = generate_matrix_elements_cython(
            np.array(range(nstrings), dtype=np.int32), strings, min_ld, max_ld
        )

    return mat


def generate_graph(strings, min_ld=1, max_ld=1, sc=None):
    """
    Generate a distance matrix and produce a graph object 

    Inputs
    ------

    strings : list
        a list of strings to use for the pairwise distance matrix
    min_ld : int, optional
        minimum Levenshtein distance
    max_ld : int, optional
        maximum Levenshtein distance
    sc : pyspark.SparkContext
        a live SparkContext; if none is given, the calculation is done locally

    Returns
    -------
    g : networkx.Graph object with strings used as label names

    """
    mat = distance_matrix(strings, sc=sc, min_ld=min_ld, max_ld=max_ld)

    if sc is not None:
        mat_data = np.array(mat.collect())
    else:
        mat_data = np.array(list(mat))

    if len(mat_data) > 0:
        comb_matrix = csr_matrix(
            (mat_data[:, 2], (mat_data[:, 0], mat_data[:, 1])),
            (len(strings), len(strings)),
        )
        # get the graph from the distance matrix
        g = nx.Graph(comb_matrix)
        string_map = {i: s for i, s in enumerate(strings)}
        nx.relabel_nodes(g, string_map, copy=False)
    else:
        g = nx.Graph()

    return g


def generate_spark_graph(strings, sc, mat=None, min_ld=1, max_ld=1):
    """
    Make a graph using the Spark graphframes library

    Inputs
    ------

    strings: list
        a list of strings to use for the pairwise distance matrix
    sc : pyspark.SparkContext
        a live SparkContext
    mat : pyspark.RDD, optional
        an RDD representing the distance matrix (returned by `distance_matrix`). If not given, 
        it is generated automatically
    min_ld : int, optional
        minimum Levenshtein distance
    max_ld : int, optional
        maximum Levenshtein distance

    Returns
    -------
    g : graphframes.GraphFrame object with strings as node names

    """
    try:
        import findspark

        findspark.init()
        import graphframes
        from pyspark.sql import Row, SQLContext
        from pyspark.sql.types import (
            StructField,
            StructType,
            IntegerType,
            ShortType,
            StringType,
            LongType,
        )
    except:
        warn("Problem importing pyspark -- are you sure your SPARK_HOME is set?")

    sqc = SQLContext(sc)

    strings_b = sc.broadcast(strings)
    size = len(strings)

    # make the vertex DataFrame
    v_schema = StructType(
        [StructField("id", IntegerType()), StructField("string", StringType())]
    )
    v_rdd = sc.parallelize(range(size)).map(
        lambda x: Row(id=x, string=strings_b.value[x])
    )
    v = sqc.createDataFrame(v_rdd, schema=v_schema)

    # make the edge DataFrame
    if mat is None:
        mat = distance_matrix(strings, min_ld=min_ld, max_ld=max_ld, sc=sc)
    e_schema = StructType(
        [
            StructField("src", IntegerType()),
            StructField("dst", IntegerType()),
            StructField("weight", ShortType()),
        ]
    )
    e = sqc.createDataFrame(mat, schema=e_schema)
    gf = graphframes.GraphFrame(v, e)

    return gf


def degree_df(strings, gf):
    """
    Create a pyspark.sql.DataFrame with a column representing the degree of each node

    Inputs
    ------

    gf : pyspark.sql.GraphFrame 
        a graph created via `produce_spark_graph`
    strings : list
        a list of strings
    
    Returns
    -------

    df : pyspark.sql.DataFrame
        a spark DataFrame with id, string, and degree columns

    """
    strings_b = sc.broadcast(strings)
    d_df = gf.degrees
    return (
        d_df.withColumn("string", d_df.id)
        .rdd.map(lambda r: Row(id=r.id, degree=r.degree, string=strings_b.value[r.id]))
        .toDF()
    )


def get_degrees(idxs, strings, min_ld, max_ld):
    """Generate a 2D array of shape (N_vertices, max_ld - min_ld + 1) which 
    holds the degree connectedness information for each node.

    This is a bit overly complicated because the numpy operation of the sort

    array[index1, index2] += 1

    turns out to be very slow. We try to get around this by building lists 
    and adding to the matrix in bigger chunks which is much more efficient. 


    Inputs: 

    idxs: list of indices

    string_map: strings mapping dictionary

    min_ld: minimum Levenshtein distance

    max_ld: maximum Levenshtein distance

    Note that the min and max Levenshtein distances are inclusive.

    Returns: 


    """
    import numpy as np
    from collections import defaultdict

    size = len(strings)
    degrees = np.zeros((size, max_ld - min_ld + 2), dtype="int32")

    deg_dict = defaultdict(list)

    for i, my_idx in enumerate(idxs):
        my_string = strings[my_idx]

        my_degrees = []

        # generate connections
        for idx in xrange(my_idx):
            ld = Levenshtein.distance(my_string, strings[idx])

            # if it's a connection, bump the degree count for source and destination
            if ld <= max_ld and ld >= min_ld:
                my_degrees.append(ld)
                deg_dict[idx].append(ld)

        deg_dict[my_idx] += my_degrees

        # save the degree count for my_idx
        if i % 100 == 0:
            for idx in deg_dict.keys():
                degree_count = np.bincount(deg_dict[idx])
                degrees[idx, 0 : len(degree_count)] += degree_count

            # reset the dictionary
            deg_dict = defaultdict(list)

    # clear the remaining items from temporary dictionary
    for idx in deg_dict.keys():
        degree_count = np.bincount(deg_dict[idx])
        degrees[idx, 0 : len(degree_count)] += degree_count

    return degrees


def get_degrees_wrapper(*args):
    """Wrapper to enable passing of get_degrees to mapPartitions"""
    res = get_degrees_cython(*args)
    yield res


def iadd(a, b):
    """Helper function for inplace add"""
    a += b
    return a


def generate_degrees(strings, min_ld=1, max_ld=1, sc=None):
    """
    Generate a matrix of degrees for the entire dataset

    Each row of the matrix corresponds to: (string, d1, d2 ... dn), where d1...dn correspond to the 
    number of connections per Levenshtein distance for the given string. 

    Inputs
    ------
    strings : list
        a list of strings to use for the pairwise distance matrix
    
    See `distance_matrix` for a description of the optional keywords. 
    """
    nstrings = len(strings)
    logger.info("number of strings " + str(nstrings))

    idxs = range(nstrings)

    if sc is not None:
        strings_b = sc.broadcast(strings)
        shuffle(idxs)
        npartitions = sc.defaultParallelism * 5
        return (
            sc.parallelize(idxs, npartitions)
            .mapPartitions(
                lambda iterator: get_degrees_wrapper(
                    np.fromiter(iterator, dtype=np.int32),
                    strings_b.value,
                    min_ld,
                    max_ld,
                )
            )
            .treeReduce(iadd, 4)
        )
    else:
        return get_degrees_cython(
            np.array(idxs, dtype=np.int32), strings, min_ld, max_ld
        )


def gene_pair_graphs(filename, sc=None):
    """Given a filename of CSV gene-pair sequence string data, group the strings by gene pair
    and produce a single-mutation graph.
    
    If there are more than 1000 strings for the gene pair, Spark can be used to speed up processing
    if a SparkContext is passed in.
    
    Input:
    
    filename: the path to the csv data file
    
    Optional Keywords: 
    
    sc: the SparkContext to use if number of strings > 1000
    """

    def keyfunc(x):
        return ", ".join(x[:2])

    with open(filename) as f:
        reader = csv.reader(open(filename))
        data = list(reader)
        data.sort(key=keyfunc)
        gb = itertools.groupby(data, lambda x: "_".join(x[:2]))

    graphs = {}

    for genes, str_iter in gb:
        strings = _extract_strings(str_iter)
        if len(strings) > 10000:
            g, mat = distance_matrix(strings, sc=sc)
        else:
            g, mat = distance_matrix(strings)

        if len(g) > 0:
            graphs[genes] = g

    return graphs


def process_file(
    input,
    kind="degrees",
    outdir="./",
    min_ld=1,
    max_ld=1,
    string_loc=None,
    sc=None,
    **kwargs
):
    """
    Read strings from a file and write degrees and/or graph to disk.

    Inputs
    ------

    filename : string
        
    """
    if isinstance(input, str):
        f = open(input, "r")
        filename = input
    else:
        f = input
        filename = input.name
    if string_loc is None:

        # try to guess the format
        l = f.readline()
        if len(l.split(",")) == 3:
            string_loc = 2
        elif len(l.split(",")) == 1:
            string_loc = 0
        f.seek(0)

    strings = list(pd.read_csv(f, header=None)[string_loc])

    if kind not in {"graph", "degrees", "all"}:
        raise RuntimeError("`kind` must be one of 'graph', 'degrees', or 'all'")

    if kind == "all":
        kind = ["degrees", "graph"]
    else:
        kind = [kind]

    outfile = os.path.join(outdir, os.path.basename(filename))

    for k in kind:
        write_output(strings, k, outfile, min_ld, max_ld, sc)


def write_output(strings, kind, outfile, min_ld=1, max_ld=1, sc=None):
    """
    Write either graph or csv degrees file to disk

    Inputs
    ------    
    strings : list
        a list of strings to use for the pairwise distance matrix
    kind : string
        one of {'degrees', 'graph'}
    outfile : filename
        output filename
    min_ld : int, optional
        minimum Levenshtein distance
    max_ld : int, optional
        maximum Levenshtein distance
    sc : pyspark.SparkContext
        a live SparkContext; if none is given, the calculation is done locally

    See `imnet.process_strings` for optional keywords
    """

    if kind == "graph":
        if not outfile.endswith(".gml"):
            outfile += ".gml"
        g = generate_graph(strings, min_ld, max_ld, sc)
        nx.write_gml(g, outfile)

    elif kind == "degrees":
        if not outfile.endswith(".degrees.csv"):
            outfile += ".degrees.csv"
        degrees = generate_degrees(strings, min_ld, max_ld, sc)
        df = pd.DataFrame(degrees[:, 1:], columns=range(1, degrees.shape[1]))
        df["string"] = strings
        df[["string"] + range(1, degrees.shape[1])].to_csv(outfile, index=False)

    else:
        raise RuntimeError("'kind' must be one of {'degrees', 'graph'}")
