# imNet: a Sequence Network Construction Toolkit

`imNet` is a software package for the generation and analysis of large-scale immunological and biological sequence networks. Used together with Apache Spark running on a computer cluster, `imNet` can be used to analyze network properties from samples of hundreds of thousands or even millions of sequences in a reasonable amount of time. 

## Installation

### pip

The simplest way to install imnet is with `pip`: 

```
$ pip install imnet
```

In addition to installing the `imnet` python library and its dependencies, this will also install the `imnet-analyze` script into your python `bin` directory. 

### From source

Make sure you have all the dependencies installed -- see [Dependencies](#dependencies) below. 

Clone the repository and install: 

```
$ git clone https://github.com/rokroskar/imnet.git
$ cd imnet
$ python setup.py install
```

If you make changes to the cython code, you will need `cython` and a usable C compiler. 

## Dependencies
You only need to worry about installing these by hand if you are installing from source or want to do development. If you just want to run `imnet`, installing it with `pip` [(see above)](#pip) should take care of the dependencies for you. 

### basic python libraries

These should all be installable via `pip` or `conda`:
* click 
* findspark 
* python-Levenshtein 
* scipy 
* networkx 
* pandas
* cython (optional)

### Spark

If your goal is to analyze large samples (> 10k strings) then distributing the computation is strongly advised. `imnet` currently does this using the [Apache Spark](http://spark.apache.org) distributed computation framework. We won't go into the details of installing and running spark here; you can [download it](spark-2.0.2-bin-hadoop2.7.tgz), and unpack the archive somewhere. At the very minimum, you need to set the `SPARK_HOME` environment variable to point to the directory where you unpacked spark 

```
$ export SPARK_HOME=/path/to/spark
```

#### Running `imnet` with Apache Spark

If you are running `spark` on a cloud resource, please refer to the [official spark documentation] for instructions on how to start up a spark cluster. To allow `imnet` to run via `spark` you will need to provide the spark URL of the `spark-master`. 

If your resource is an academic HPC (high-perfomance computing) cluster, we recommend that you use [`sparkhpc`](https://github.com/rokroskar/sparkhpc) for managing spark clusters. `sparkhpc` greatly simplifies spawning and managing spark clusters. 

## Basic usage

Refer to the command-line help for usage: 

```
$ imnet-analyze --help
Usage: imnet-analyze [OPTIONS] COMMAND [ARGS]...

Options:
  --spark-config TEXT         Spark configuration directory
  --spark-master TEXT         Spark master
  --kind [graph|degrees|all]  Which kind of output to produce
  --outdir TEXT               Output directory
  --min-ld INTEGER            Minimum Levenshtein distance
  --max-ld INTEGER            Maximum Levenshtein distance
  --sc-cutoff INTEGER         For number of strings below this, Spark will not
                              be used
  --spark / --no-spark        Whether to use Spark or not
  --help                      Show this message and exit.

Commands:
  benchmark  Run a series of benchmarks for graph and...
  directory  Process a directory of CDR3 string files
  file       Process an individual file with CDR3 strings...
  random     Run analysis on a randomly generated set of...

$ imnet-analyze random --help
Usage: imnet-analyze random [OPTIONS]

  Run analysis on a randomly generated set of strings for testing

Options:
  --nstrings INTEGER    Number of strings to generate
  --min-length INTEGER  minimum number of characters per string
  --max-length INTEGER  maximum number of characters per string
  --help                Show this message and exit.
```

For an example of using the `imnet` python library, have a look at the [example notebook](notebooks/example_workflow.ipynb).



