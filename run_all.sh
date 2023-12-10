# This script runs all the experiments performed for BioMON. It is recommended to run this script on a server with a GPU.
# Authors: Manos Chatzakis (emmanouil.chatzakis@epfl.ch), Lluka Stojollari (lluka.stojollari@epfl.ch)


run_bioMON_experiments() {
    dataset_name=$1
    echo "Dataset: $dataset_name"

    for classifier in "SVM" "LR" "MLP"
    do
        model_name=bioMON_${classifier}
        #echo "Running BioMON experiments for: $model_name"

        echo python3 run.py exp.name=final method=$model_name dataset=$dataset_name 
    done

} 

run_competitor_experiments() {
    dataset_name=$1
    echo "Dataset: $dataset_name"

    for classifier in "SVM" "LR" "MLP"
    do
        model_name=bioMON_${classifier}
        #echo "Running BioMON experiments for: $model_name"

        echo python3 run.py exp.name=final method=$model_name dataset=$dataset_name 
    done

}





#
# Part1: FCNET Backbone
#
# 

echo "Running experiments with FCNET backbone"
run_bioMON_experiments "tabula_muris"

# BioMON Experiments


# Competitor Experiments




# RESNET

# R2D2