# This script runs all the experiments performed for BioMON. It is recommended to run this script on a server with a GPU.
# Authors: Manos Chatzakis (emmanouil.chatzakis@epfl.ch), Lluka Stojollari (lluka.stojollari@epfl.ch)
# To make this script runnable: $chmod u+x run_all.sh

N_WAY=5
N_SHOT=5
N_QUERY=15
EPOCHS=30
EPISODES=50

echo "========= BioMON Experiment Script ========="
echo ">> This script runs all the experiments performed for BioMON. It is recommended to run this script on a server with a GPU."
echo ">> Authors: Manos Chatzakis (emmanouil.chatzakis@epfl.ch) Lluka Stojollari (lluka.stojollari@epfl.ch)"
echo ">> Note: Make sure you have Swissprot downloaded and unzipped in the data folder."
echo ">> Epochs: $EPOCHS, Episodes: $EPISODES, N-way: $N_WAY, N-shot: $N_SHOT, N-query: $N_QUERY"
echo ""

run_benchmark_algorithms(){
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for method in "maml" "protonet" "matchingnet" "baseline" "baseline_pp"
    do
        model_name=${method}.yaml
        python3 run.py exp.name=final \
            method=$model_name \
            model=$backbone_name \
            dataset=$dataset_name \
            backbone._target_=$backbone_target \
            backbone.layer_dim=$layer_dim \
            n_way=$N_WAY \
            n_shot=$N_SHOT \
            n_query=$N_QUERY \
            iter_num=$EPISODES \
            method.stop_epoch=$EPOCHS \
            method.start_epoch=0     
    done
}

run_bioMON_simple_classifiers(){
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for classifier in "SVM" #"LR" "DT" "NB" "GMM" 
    do
        model_name=bioMON_${classifier}.yaml
        
        python3 run.py exp.name=final \
            method=$model_name \
            model=$backbone_name \
            dataset=$dataset_name \
            backbone._target_=$backbone_target \
            backbone.layer_dim=$layer_dim \
            n_way=$N_WAY \
            n_shot=$N_SHOT \
            n_query=$N_QUERY \
            iter_num=$EPISODES \
            method.stop_epoch=$EPOCHS \
            method.start_epoch=0 

    done
}

run_bioMON_KNN() {
    
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for classifier in "1NN" "2NN" "3NN" "4NN" "5NN"
    do
        model_name=bioMON_${classifier}.yaml
        
        python3 run.py exp.name=final \
            method=$model_name \
            model=$backbone_name \
            dataset=$dataset_name \
            backbone._target_=$backbone_target \
            backbone.layer_dim=$layer_dim \
            n_way=$N_WAY \
            n_shot=$N_SHOT \
            n_query=$N_QUERY \
            iter_num=$EPISODES \
            method.stop_epoch=$EPOCHS \
            method.start_epoch=0 

    done
} 

run_bioMON_RF() {
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for classifier in "RF10" "RF50" "RF100" "RF200"
    do
        model_name=bioMON_${classifier}.yaml
        
        python3 run.py exp.name=final \
            method=$model_name \
            model=$backbone_name \
            dataset=$dataset_name \
            backbone._target_=$backbone_target \
            backbone.layer_dim=$layer_dim \
            n_way=$N_WAY \
            n_shot=$N_SHOT \
            n_query=$N_QUERY \
            iter_num=$EPISODES \
            method.stop_epoch=$EPOCHS \
            method.start_epoch=0 

    done
}

run_bioMON_MLP() {
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for classifier in ""
    do
        model_name=bioMON_${classifier}.yaml
        
        python3 run.py exp.name=final \
            method=$model_name \
            model=$backbone_name \
            dataset=$dataset_name \
            backbone._target_=$backbone_target \
            backbone.layer_dim=$layer_dim \
            n_way=$N_WAY \
            n_shot=$N_SHOT \
            n_query=$N_QUERY \
            iter_num=$EPISODES \
            method.stop_epoch=$EPOCHS \
            method.start_epoch=0 

    done
}

fcnet_target=backbones.fcnet.FCNet
fcnet_name=FCNet

echo "========= Running all experiments for Swissprot ========="
fcnet_layer_dim=[512,512]
# run_benchmark_algorithms "swissprot" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_simple_classifiers "swissprot" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_KNN "swissprot" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_RF "swissprot" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_MLP "swissprot" $fcnet_name $fcnet_target $fcnet_layer_dim
echo ""

echo "========= Running all experiments for tabula_muris ========="
fcnet_layer_dim=[64,64]
# run_benchmark_algorithms "tabula_muris" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_simple_classifiers "tabula_muris" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_KNN "tabula_muris" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_RF "tabula_muris" $fcnet_name $fcnet_target $fcnet_layer_dim
# run_bioMON_MLP "tabula_muris" $fcnet_name $fcnet_target $fcnet_layer_dim
echo ""

echo ""
echo "========= Script completed. Reporting. ========="
echo ">> The results of all experiments are placed under ./results/final/"
echo ">> To generate the graphs, run the notebook bioMON.ipynb"

