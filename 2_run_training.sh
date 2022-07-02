#!/bin/bash

#inputFileName="protons.root"
inputFileName="protons_b2.root"

#class_methods="PDERSPCA,PDEFoamBoost,KNN,BoostedFisher,MLP,SVM,BDT,RuleFit,DNN_CPU"
#outputFileName="tmva.root"
#outputFileName="tmva_knn_svm_bdt.root"

#class_methods="KNN,SVM,BDT,BDTG,LikelihoodPCA,PDERSPCA,PDEFoamBoost,BoostedFisher,MLP,MLPBNN,RuleFit,DNN_CPU"
#outputFileName="tmva_many.root"

#class_methods="BDT,BDTG,BDTB,BDTD,BDTF"
#outputFileName="tmva_bdts.root"

#class_methods="Cuts,CutsD,CutsPCA,CutsGA,CutsSA"
#outputFileName="tmva_cuts.root"

#class_methods="Likelihood,LikelihoodD,LikelihoodPCA,LikelihoodKDE,LikelihoodMIX"
#mix not okay 
#class_methods="Likelihood,LikelihoodD,LikelihoodPCA,LikelihoodKDE"
#outputFileName="tmva_likelihoods.root"

#class_methods="PDERS,PDERSD,PDERSPCA,PDEFoam,PDEFoamBoost,KNN"
#outputFileName="tmva_mlikelihoods_knn.root"

#class_methods="LD,Fisher,FisherG,BoostedFisher,HMatrix"
#outputFileName="tmva_lda.root"

#class_methods="FDA_GA,FDA_SA,FDA_MC,FDA_MT,FDA_GAMT,FDA_MCMT"
#outputFileName="tmva_lda.root"

#class_methods="MLP,MLPBNN,CFMlpANN,TMlpANN,DNN_CPU"
#MLPBFGS not working
#outputFileName="tmva_mlps.root"

#class_methods="SVM,RuleFit"
#outputFileName="tmva_svm_rulefit.root"

#class_methods="BDT,BDTG,BDTB,BDTD,BDTF"
#outputFileName="tmva_bdts.root"

class_methods="CutsSA,LikelihoodKDE,KNN,PDEFoam,FisherG,TMlpANN,RuleFit,BDT,SVM"
#Performance of SVM is not good but put here as a reference
#outputFileName="tmva_star_cast.root"
outputFileName="tmva_star_cast_b2.root"

exe_tmva_str="root -b -q 'TMVAClassification.C(\""$class_methods\"", \""$inputFileName\"", \""$outputFileName\"")'"

#Run training code
echo "Run TMVA training code ..." $exe_tmva_str" ......"
eval $exe_tmva_str
