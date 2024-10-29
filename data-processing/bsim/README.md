# BSim Processing

## Setup

1. Install [Ghidra](https://ghidra-sre.org) 

2. Create a BSim Server (Postgres is recommended, though the setup is more intensive)

3. Create a BSim database for the dataset you're analyzing

```<ghidra base dir>/support/bsim createdatabase ...```

4. Fill out config in `.env-blank` and rename to `.env`

5. Run the evaluation on the dataset like this:

```python src/pipeline.py /path/to/binaries/directory```

The pipeline will create the signatures, commit them to the database, query for function rankings, and calculate MRR. Results will be saved in the time-logging file. By default, the multithreading will use a number of workers equal to the number of cores minus one. 

If you only want to run the pipeline on a subset of binaries, or in binaries scattered across disparate folders, you can provide a `.txt` file with a new-line separated list of absolute paths, like so:

```python src/pipeline.py /path/to/binaries/list.txt```

The full pipeline might take a long time depending on the amount of compute resources available.