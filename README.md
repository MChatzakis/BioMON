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

## Usage
Here we provide important information of how the project is organized, and how to use the provided code.

### Organization
The repository uses hydra to operate. All the configurations needed, included hyperparameters and which methods are used can be found under the conf/ directory. 

### Provided Methods
We provide various classifiers available, from classic ML methods (Logistic Regression, SVMs, ...) to Deep Learning Neural Networks. All classifiers are available at [heads.py](aaa). BioMON supports two embedding methods, [FCNet](cite) and and [R2D2](cite), available under backbones/ directory. In addition, we provide other few-shot learning methods serving as competitors, summarized below:

| Method      | Source                             | 
|--------------|----------------------------------|
| Baseline, Baseline++ | [Chen et al. (2019)](https://arxiv.org/pdf/1904.04232.pdf) |
| ProtoNet | [Snell et al. (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) |
| MatchingNet | [Vinyals et al. (2016)](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf) |
| MAML | [Finn et al. (2017)](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) |

### Custom Training and Testing
To directly test or train BioMON or any of the competitors, the [run.py](aa) file shall be used. To run it, use:
```bash
python3 run.py exp.name={name} \
            method={method} \
            model={backbone_name} \
            dataset={dataset} \
            backbone._target_={backbone_class}\
            backbone.layer_dim={backbone_layers} \
            n_way={n_way} \
            n_shot={n_way} \
            n_query={n_way} \
            iter_num={n_way} \
            method.stop_epoch={n_way} \
            method.start_epoch={n_way}     
```
In case any of those parameters are not used, the default parameters (found in corresponding files of conf/ directory will be used).
An example of the run is
```bash
python3 run.py exp.name={name} \
            method={method} \
            model={backbone_name} \
            dataset={dataset} \
            backbone._target_={backbone_class}\
            backbone.layer_dim={backbone_layers} \
            n_way={n_way} \
            n_shot={n_way} \
            n_query={n_way} \
            iter_num={n_way} \
            method.stop_epoch={n_way} \
            method.start_epoch={n_way}     
```


## About
This project was developed for the [Deep Learning in Biomedicine course of EPFL (cs503)](aa). The episodic data loaders used to load the datasets were provided by the course, while the competitor algorithms are adaptions of the online versions available. 