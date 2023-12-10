# This script runs all the experiments performed for BioMON. It is recommended to run this script on a server with a GPU.
# Authors: Manos Chatzakis (emmanouil.chatzakis@epfl.ch), Lluka Stojollari (lluka.stojollari@epfl.ch)
# To make this script runnable: $chmod u+x run_all.sh

N_WAY=5
N_SHOT=5
N_QUERY=15
EPOCHS=5
EPISODES=2

echo "========= BioMON Experiment Script ========="
echo ">> This script runs all the experiments performed for BioMON. It is recommended to run this script on a server with a GPU."
echo ">> Authors: Manos Chatzakis (emmanouil.chatzakis@epfl.ch) Lluka Stojollari (lluka.stojollari@epfl.ch)"
echo ">> Note: Make sure you have Swissprot downloaded and unzipped in the data folder."
echo ">> Epochs: $EPOCHS, Episodes: $EPISODES, N-way: $N_WAY, N-shot: $N_SHOT, N-query: $N_QUERY"
echo ""

run_experiments() {
    dataset_name=$1
    backbone_name=$2
    backbone_target=$3
    layer_dim=$4

    echo "  Dataset: $dataset_name, Backbone: ($backbone_target, $layer_dim)"

    for classifier in "DT" "GMM" "KNN" "LR" "MLP" "NB" "RF" "RR" "SVM" 
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

    for method in "maml" "protonet" "matchingnet" "baseline" "baseline_pp"
    do
        model_name=${method}.yaml
        echo python3 run.py exp.name=final method=$model_name dataset=$dataset_name backbone._target_=$backbone_target backbone.layer_dim=$layer_dim
    done

} 

# Start the experiments
echo "========= Running all experiments ========="

#
# Part1: FCNET Backbone
#
# 
fcnet_target=backbones.fcnet.FCNet
fcnet_layer_dim=[512,512]

run_experiments "tabula_muris" "FCNET" $fcnet_target $fcnet_layer_dim
#run_experiments "swissprot" $fcnet_target $fcnet_layer_dim

echo ">>FCNET Backbone experiments completed."



echo ""
echo "========= Script completed. Reporting. ========="
echo ">> The results of all experiments are placed under ./results/final/"
echo ">> To generate the graphs, run the notebook bioMON.ipynb"

