# Tentative pipeline:
1. Use TimeNorm to normalize times

    a. in one terminal, `cd` to `scala_time_normer` and run `sbt run`

	b. in another, run `get_normed_times.py` with the appropriate arguments
2. post-process the normed times

	a. this will be incorporated into `get_normed_times.py` later on
3. run `augment_graphs_with_normed_times.py` with the appropriate arguments to create graphs where normed timexes are a separate class

For example:

```
cd scala_time_normer
sbt run
```
```
python get_normed_times.py -g /path/to/your/graphs -o /path/to/your/normed_times -t /path/to/your/discharge_dates.json
```

You can interrupt the scala program once this is done.

```
python postprocess_normed_times.py -n /path/to/your/normed_times -o /path/to/your/processed_normed_times -t /path/to/your/discharge_dates.json
```
```
python augment_graphs_with_normed_times.py -g /path/to/your/graphs -o /path/to/your/final_graphs -t /path/to/your/discharge_dates.json
```
