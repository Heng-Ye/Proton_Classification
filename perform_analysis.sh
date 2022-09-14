#!/bin/bash

#Setup Features that will NOT be used ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#feature_obs_not_be_used="'train','tag','target'"

#feature_obs_not_be_used="'Etrklen','keffbeam','keffhy','kendbb','kendfitbb'"
#feature_obs_not_be_used="'keffbeam','keffhy','kendbb','kendfitbb'"
#feature_obs_not_be_used="'Etrklen','keffhy','kendbb','kendfitbb'"
#feature_obs_not_be_used="'Etrklen','keffbeam','kendbb','kendfitbb'"
#feature_obs_not_be_used="'Etrklen','keffbeam','keffhy','kendfitbb'"
#feature_obs_not_be_used="'Etrklen','keffbeam','keffhy','kendbb'"
feature_obs_not_be_used="'non'"
#feature_obs_not_be_used="'ntrklen','trklen','Etrklen','PID','B','costheta','mediandedx','endpointdedx','calo','avcalo','keffbeam','keffhy','kendbb','kendfitbb','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd'"
#feature_obs_not_be_used="'trklen','Etrklen','B','costheta','mediandedx','endpointdedx','calo','avcalo','keffbeam','keffhy','kendbb','kendfitbb','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd'"
#feature_obs_not_be_used="'Etrklen','keffbeam','keffhy','kendbb','kendfitbb','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd'"
#feature_obs_not_be_used="'end_x','end_y','end_z','st_z',nd'"

#feature_obs_not_be_used="'train','tag','target'"
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#I/O ------------------------------------------------
input_file="./data_prep/protons_mva2"
output_xgb_model="./models/xgb_mva2.model"
plt_path_xgbm="./plts_perform_mva2_xgbm/"
output_inel_csv="./csv_mva2/xgb_inel_xgb_valid.csv"

output_lgbm_model="./models/lgbm_mva2.model"
plt_path_lgbm="./plts_perform_mva2_lgbm/"
output_inel_lgbm_csv="./csv_mva2/xgb_inel_lgbm_valid.csv"


#input_file="./data_prep/protons_mva3"
#output_xgb_model="./models/xgb_mva3_basic.model"
#plt_path_xgbm="./plts_perform_mva3_xgbm/"
#output_inel_csv="./csv_mva3/xgb_inel_xgb_valid.csv"


#output_lgbm_model="./models/lgbm_mva3.model"
#plt_path_lgbm="./plts_perform_mva3_lgbm/"
#output_inel_lgbm_csv="./csv_mva3/xgb_inel_lgbm_valid.csv"

plt_path_com="./plts_perform_mva2_combine/"

#----------------------------------------------------

#Run Analysis -----------------------------------------------------------------------------------------------------------------------
#python ml_1_data_prep.py -d $input_file -f $feature_obs_not_be_used
#python ml_1_data_prep.py -d $input_file
#python ml_2_train_xgboost.py -d $input_file -f $feature_obs_not_be_used -o $output_xgb_model
#python ml_3_evaluate_xgboost.py -d $input_file -f $feature_obs_not_be_used -o $output_xgb_model -p $plt_path_xgbm -ocsv $output_inel_csv

python ml_2_train_lightgbm.py -d $input_file -f $feature_obs_not_be_used -o $output_lgbm_model -p $plt_path_lgbm
python ml_3_evaluate_lightgbm.py -d $input_file -f $feature_obs_not_be_used -o $output_lgbm_model -p $plt_path_lgbm -ocsv $output_inel_lgbm_csv

#python ml_3_combine_lightgbm_xgboost.py -d $input_file -f $feature_obs_not_be_used -o1 $output_lgbm_model -o2 $output_xgb_model -p $plt_path_com


