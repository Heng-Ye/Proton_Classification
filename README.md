# Proton Classification using Clustering/Regression/Deep Neural Network/MLP

Code:<br/>
-Produce new tree for training\
├── Tree Maker: **ProtonNewTreeMakerX.C** \
├── Run "./1_create_tree.sh" to execute the code
 
-Run training process\
├── Training/classification code: **TMVAClassification.C** \
├── Choose the classification methods and execute the training process by "./2_run_training.sh" 

-Use TMVA GUI to see the trained results\
├── Open the TMVA GUI by 'root -l'\
├── **TMVA::TMVAGui(**“TMVA.root”**)**\
├── Select "ROC" curve for the trained performance\
├── After selecting the ROC curve, the weighted results will be altomatically generated

-Use the trained results and apply on data\
├── Main code: **TMVAClassificationApplication.C**\
├── Run "./3_apply_cut.sh"
