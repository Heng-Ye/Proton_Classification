# Proton Event Selection using Advanced Machine Learning Regression/Classification Methods

Framework:<br/>
-Produce new tree for training\
├── (1) Tree Maker: **ProtonNewTreeMakerX.C** \
├── (2) Run "./1_create_tree.sh" to execute the code \
├── (3) Root-CSV conversion: **root2csv_converter.C** 
 \\

-**Regression using XGBoost, LightGBM**\
├── (1) Data preparation code: ml_1_data_prep.py (Randomly divide data into training and validation(test) sets   
├── (2) Training using modern models (XGBoost, LightGBM): ml_2_train_xgboost.py, ml_2_train_lightgbm.py \
├── (3) Performance evaluation code: ml_3_evaluate_xgboost.py, ml_3_evaluate_lightgbm.py           
├── (4) **Data visualization/Dimensionality reduction using UMAP**: ml_4_feature_visualization.py
 \

-**ML Classification using KNN, SVM, MLP, Likelihood PCA, BoostedFisher (HEP-TMVA package)** \ 
-Run training process \
├── Training/classification code: **TMVAClassification.C** 
├── Choose the classification methods and execute the training process by "./2_run_training.sh" 

 \

-Use TMVA GUI to see the trained results\
├── Open the TMVA GUI by 'root -l'\
├── **TMVA::TMVAGui(**“TMVA.root”**)**\
├── Select "ROC" curve for the trained performance\
├── After selecting the ROC curve, the weighted results will be altomatically generated

-Use the trained results and apply on data\
├── Main code: **TMVAClassificationApplication.C**\
├── Run "./3_apply_cut.sh"
