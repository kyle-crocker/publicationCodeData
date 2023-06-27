This repository contains code and data associated with the manuscript entitled "Genomic patterns in the global soil microbiome emerge from microbial interactions" by Kyle Crocker, Milena Chakraverti-Wuerthwein, Zeqian Li, Madhav Mani, Karna Gowda, and Seppe Kuehn. 

All python code uses version 3.9.1, numpy version 1.21.1, and pandas version 2.0.2, except for code contained in FigS8_coevolution_code_data, which uses python version 3.11.0, numpy version 1.23.5, and pandas version 1.5.2. Matlab code uses version 2017b. 

Code is commented to provide instructions for its use. 

Expected output of the code is the indicated figure in the manuscript. 

All scripts should run in < 1 minute except for fit_and_predict_code_data/CRM_predict.ipynb, which should run in < 15 minutes, and simulation_figure_code_data/num_roots_k_I_tox_min_full.py, which depends on the command line arguments. It should take < 2 hours to reproduce Fig. S9. 
