How to run an algorithm on a dataset:

```python
python code/run.py config_file
```

In addition to the hyperparameters of the selected algorithm, `config_file` should contain the following:

- `data=path/to/data.csv`, a .csv file containing the data, one vector per line; or, for multiview learning, `data=path/to/view1.csv,path/to/view2.csv,path/to/view3.csv`

- `algorithm=name_of_algorithm`, for a list see `Algorithms/__init__.py`

- `savefile=path/to/where/the_final_parameters/will_be_saved.pkl`, a file where the instance of the algorithm will be saved in pickle format
