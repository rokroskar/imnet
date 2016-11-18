# AbNet: a CDR3 Network Analysis Toolkit

## Installation
```
$ python setup.py install
```

In addition to installing the `abnet` python library, this will also install the `abnet-analyze` script into your python `bin` directory. 

## Dependencies
See `requirements.txt` for details. Briefly: 

* `python_Levenshtein`: python wrapper around a C-based Levenshtein library
* scientific python stack: `scipy`, `numpy`
* `click` command-line-interface library 
* graph libraries: at least `networkx`, but can also use `igraph`, `graph_tool` for better performance
* [`spark`](http://spark.apache.org): for distributing the computation. Advised for samples with > 10k strings
* [`sparkhpc`](https://github.com/rokroskar/sparkhpc): for managing spark clusters on an HPC resource

## Usage

Refer to the command-line help for usage: 

```
$ abnet-analyze --help
Usage: abnet-analyze [OPTIONS] COMMAND [ARGS]...

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

$ abnet-analyze random --help
Usage: abnet-analyze random [OPTIONS]

  Run analysis on a randomly generated set of strings for testing

Options:
  --nstrings INTEGER    Number of strings to generate
  --min-length INTEGER  minimum number of characters per string
  --max-length INTEGER  maximum number of characters per string
  --help                Show this message and exit.
```

For an example of using the `abnet` python library, have a look at the [example notebook](notebooks/example_workflow.ipynb).



