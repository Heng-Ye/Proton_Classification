#!/bin/bash

#class_methods="CutsSA,LikelihoodKDE,KNN,FisherG,TMlpANN,RuleFit,BDT,SVM"
class_methods="BDT"

#Source file to apply cut
inputSourceFileName="protons_mva2.root"

#traing methods
#inputTrainingFileName="tmva_star_cast_mva2.root"

#save results after TMVA cuts
OutputFileName="tmva_app_bdt_mva2.root"



exe_tmva_str="root -b -q 'TMVApplication.C(\""$class_methods\"", \""$inputSourceFileName\"", \""$outputFileName\"")'"

#Run training code
echo "Run TMVA training code ..." $exe_tmva_str" ......"
eval $exe_tmva_str
