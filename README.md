# BioMON: Biomedical Few-shot Meta-learning Architectures with Classification Base Learners

BioMON is a Few-shot Meta-learning architecture that employs various classifiers as base learners, targeted for biomedical data collections. Developed by Manos Chatzakis (emmanouil.chatzakis@epfl.ch) and Lluka Stojollari (lluka.stojollari@epfl.ch).

## Quick Start

We provide an environment with all the packages needed, supported using conda. Create it using
```bash
conda env create -f environment.yml
```

The environment can be activated using 
```bash
conda activate few-shot-benchmark
```

Alternatively, the needed packages can be installed using pip
```bash
python -m pip install -r requirements.txt
```

BioMON operates on Tabula Muris and Swissprot benchmarks. Tabula Muris is automatically downloaded if it is not presents in the directory. Swissprot should be downloaded from [here](aa), and placed unzipped under a data/ directory. 

The complete training of all BioMON variatons and competitor algorithms can be reproduced by running
```bash
chmod u+x run_all.sh # Makes the script executable
./run_all.sh
```
It is important to run the above script is a server with a GPU, as it needs a lot of time to complete. 

The complete experimental evaluation can be reproduced (after running run_all.sh) by running the notebook [BioMON.ipynb](aa) provided in this repository. It contains all the graphs used in the report, and additional plots that we did not include due to space constraints.  