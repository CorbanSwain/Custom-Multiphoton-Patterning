# Custom Multi-photon Patterning
This codebase is design to take image- or vector-based ''masks'' and convert
them into mask files that can be imported into 2-photon imaging software
(PrairieView by Bruker) and direct the microscope system to selectively
"expose" regions of a sample to laser power according to the mask.


## Getting Started
The easiest way to start will be to download this code, and use Anaconda 
to setup a python environment.
Then use a jupyter notebook to load and run the conversion code. You can
do this via the following steps:

1. Download a release or 
   [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) 
   this repository into a folder (e.g. `.../cmp/`) on your computer. 
   * [**Download the latest release here.**](https://github.com/CorbanSwain/Custom-Multiphoton-Patterning/releases/latest/download/cmp.zip)
   * *Note*: If you clone this repository, you will need to import the submodules
     via the following commands:
     ```bash
     git submodule init
     git submodule update
     ```
     
1. Download and install 
   [Anaconda (individual edition)](https://www.anaconda.com/products/individual)
   if you do not already have it
   
1. From the anaconda prompt `cd` to this repository directory, then run *one* 
   the following commands.
   ```bash
   conda env create --prefix ./env --file env.yml 
   ```
   If issues occur with the first method here you can try generating the 
   enviornment from the explicit package list:
   ```bash
   conda env create --prefix ./env --file .env-explicit.txt
   ```
1. Activate the anaconda enviornment by the following command
   ```bash
   conda activate ./env
   ```
   
1. Open a jupyter notebook by entering the following command
   ```bash
   jupyter lab getting_started.ipynb
   ```
   
1. Once in the jupyter notebook run and edit the cells to see how to use the 
   mask generation codebase.


## Todo
- [ ] Add more examples of using parameters.
- [ ] Add documentation to classes and functions.
- [ ] Handling additional edge cases.

### Done
- [x] Create example jupyter notebook.

