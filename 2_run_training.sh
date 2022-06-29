#!/bin/bash

inputFileName="protons.root"

#class_methods="PDERSPCA,PDEFoamBoost,KNN,BoostedFisher,MLP,SVM,BDT,RuleFit,DNN_CPU"
#outputFileName="tmva.root"
class_methods="KNN,SVM,BDT"
outputFileName="tmva_knn_svm_bdt.root"

exe_tmva_str="root -b -q 'TMVAClassification.C(\""$class_methods\"", \""$inputFileName\"", \""$outputFileName\"")'"

#Run training code
echo "Run TMVA training code ..." $exe_tmva_str" ......"
eval $exe_tmva_str
