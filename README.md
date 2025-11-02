# HPC---Parallel-Random-Forest
A parallel implementation of Random Forest (multithread/multinode).

<p align="center">
  <img src="pics/digital_forest.png" width="600" height="300">  
</p>

## How it works

1. Create a dataset
    ```
    cd data
    python -m venv hpc_venv
    source hpc_venv/bin/activate
    pip install -r requirements.txt
    python make_dataset.py  
    ```
2. Train and test
    ```
    make 
    make clean
    ./final 
    ```

If you want to run only inference then use:
```
./final --trained_tree_path random_tree.bin
```
