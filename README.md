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

## Get the Documentation

First get doxygen:

```
sudo apt install doxygen
```

Run the following in the project root to get the config file Doxyfile:

```
doxygen -g
```

Adjust the config file as needed. Then run the following to get the latex documentation:

```
doxygen Doxyfile
```

Finally, if you want a pdf you can use:
```
sudo apt install texlive
sudo apt install texlive-latex-extra
cd latex
pdflatex refman.tex
```
However, at the moment the pdf generated is not correct.