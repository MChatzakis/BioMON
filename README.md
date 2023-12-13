# BioMON: Biomedical Few-shot Meta-learning Architectures with Classification Base Learners

BioMON is a Few-shot Meta-learning architecture that employs various classifiers as base learners, targeted for biomedical data collections. 

Developed by Manos Chatzakis (emmanouil.chatzakis@epfl.ch) and Lluka Stojollari (lluka.stojollari@epfl.ch).

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

BioMON operates on Tabula Muris and Swissprot benchmarks. Tabula Muris is automatically downloaded if it is not presents in the directory. Swissprot should be downloaded from [here]([here](https://drive.google.com/drive/u/0/folders/1IlyK9_utaiNjlS8RbIXn1aMQ_5vcUy5P)), and placed unzipped under a data/ directory. 

The complete training of all BioMON variatons and competitor algorithms can be reproduced by running
```bash
chmod u+x run_all.sh # Makes the script executable
./run_all.sh
```
It is important to run the above script is a server with a GPU, as it needs a lot of time to complete. 

The complete experimental evaluation can be reproduced (after running run_all.sh) by running the notebook [bioMON.ipynb](bioMON.ipynb) provided in this repository. It contains all the graphs used in the report, and additional plots that we did not include due to space constraints.

## Usage
Here we provide important information of how the project is organized, and how to use the provided code.

### Organization
The repository uses hydra to operate. All the configurations needed, included hyperparameters and which methods are used can be found under the conf/ directory. 

### Provided Methods
We provide various classifiers available, from classic ML methods (Logistic Regression, SVMs, ...) to Deep Learning Neural Networks. All classifiers are available at [heads.py](methods/heads.py). BioMON supports two embedding methods, [FCNet](https://link.springer.com/chapter/10.1007/978-3-319-67159-8_9) and and [R2D2](https://arxiv.org/abs/1805.08136), available under backbones/ directory. In addition, we provide other few-shot learning methods serving as competitors, such as Protonet ([Snell et al. (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)), MAML ([Finn et al. (2017)](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)), MatchingNet ([Vinyals et al. (2016)](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf)), and Baselines ([Chen et al. (2019)](https://arxiv.org/pdf/1904.04232.pdf)):

### Custom Training and Testing
To directly test or train BioMON or any of the competitors, the [run.py](run.py) file shall be used. To run it, use:
```bash
python3 run.py exp.name={name} method={method} model={backbone_name} dataset={dataset} backbone._target_={backbone_class} backbone.layer_dim={backbone_layers} n_way={n_way} n_shot={n_shot} n_query={n_query} iter_num={episodes} method.stop_epoch={stop_epoch} method.start_epoch={start_epoch}     
```
In case any of those parameters are not used, the default parameters (found in corresponding files of conf/ directory will be used).


An example of the run is
```bash
python run.py exp.name=random_test method=bioMON_LR dataset=tabula_muris model=R2D2 backbone._target_=backbones.r2d2.R2D2 backbone.layer_dim=[64,64] n_way=5 n_shot=5 n_query=15 iter_num=100 method.stop_epoch=30 method.start_epoch=0 
```
The above command will run train 30 epochs of BioMON with a Logistic Regression classifier on the Tabula Muris dataset, using a 2-layer R2D2 embedding, for 5-way 15-shot learning with 15 queries per episode. The results will be saved under results/random_test/tabula_muris/.

In order to explicitely test a model (not train), an additional argument `mode=test` should be used for run.py. Also, we use Wandb for experiment tracking. To set it, see the corresponding [section](#experiment-tracking). To disable it, use `wandb.mode=disabled`

The available methods for the method argument of run.py are summarized below.

| Method      | Description                             | 
|--------------|----------------------------------|
| baseline, baseline_pp | Baseline implementations (competitors) |
| protonet | Protonet implementation (competitor) |
| matchingnet | MatchingNet implementation (competitor) |
| maml | MAML implementation (competitor) |
| bioMON_{k}NN | BioMON with KNN, for specific k value from 1-5 |
| bioMON_DT | BioMON with Decision Tree |
| bioMON_GNN | BioMON with a classification variation of Gaussian Mixture Model |
| bioMON_LR | BioMON with Logistic Regression |
| bioMON_NB | BioMON with Naive Bayes |
| bioMON_RF{n}| BioMON with a Random Forest of various estimators, specified with n, for 10,50,100,200 |
| bioMON_SVM | BioMON with SVM |
| bioMON_MLP_e{epochs}_l{layers} | BioMON with MLP Network. Epochs={1,5,10,15}, layers={128-64, 256-64-64, 512-256-128-64}

## Experiment Tracking

We use [Weights and Biases](https://wandb.ai/) (WandB) for tracking experiments and results during training. 
All hydra configurations, as well as training loss, validation accuracy, and post-train eval results are logged. For more on Hydra, see [their tutorial](https://hydra.cc/docs/intro/). For an example of a benchmark that uses Hydra for configuration management, see [BenchMD](https://github.com/rajpurkarlab/BenchMD).
To disable WandB, use `wandb.mode=disabled`. 

You must update the `project` and `entity` fields in `conf/main.yaml` to your own project and entity after creating one on WandB.

To log in to WandB, run `wandb login` and enter the API key provided on the website for your account.


## About
This project was developed for the [Deep Learning in Biomedicine course of EPFL (cs503)](https://edu.epfl.ch/coursebook/fr/deep-learning-in-biomedicine-CS-502). The episodic data loaders used to load the datasets were provided by the course, while the competitor algorithms are adaptions of the online versions available. 