# HPC---Parallel-Random-Forest
A parallel implementation of Random Forest (multithread/multinode).

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