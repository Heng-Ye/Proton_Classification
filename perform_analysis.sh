#!/bin/bash

input_file="./data_prep/protons_mva2_20220904"
#feature_obs_not_be_used="'train','tag','target'"
#feature_obs_not_be_used="'train','tag','target','ntrklen','trklen','PID','B','costheta','mediandedx','endpointdedx','calo','avcalo','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd','keffbeam','keffhy','kend_bb','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffhy_calo','r_keffhy_keffbeam'"
#feature_obs_not_be_used="'train','tag','target','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd','keffbeam','keffhy','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffhy_bb','dkeffhy_calo','r_keffhy_keffbeam'"
#feature_obs_not_be_used="'train','tag','target','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd','keffbeam','keffhy','kend_bb','kend_fit_bb','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffhy_calo','r_keffhy_keffbeam'"

feature_obs_not_be_used="'train','tag','target','st_x','st_y','st_z','end_x','end_y','end_z','pbdt','nd','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffbeam_bb','dkeffbeam_calo','dkeffhy_bb','dkeffhy_calo','r_keffhy_keffbeam'"

output_xgb_model="./models/xgb_mva2_20220904.model"
plt_path="./plts_perform_20220904/"
output_inel_csv="./csv_20220904/xgb_inel_valid.csv"


#train,tag,target,ntrklen,trklen,PID,B,costheta,mediandedx,endpointdedx,
#calo,avcalo,st_x,st_y,st_z,
#end_x,end_y,end_z,
#pbdt,nd,
#keffbeam,keffhy,
#kend_bb, dkeffbeam_bb, 
#dkeffbeam_calo, dkeffhy_bb, 
#dkeffhy_calo, r_keffhy_keffbeam

python ml_1_data_prep.py -d $input_file -f $feature_obs_not_be_used
python ml_2_train_xgboost.py -d $input_file -f $feature_obs_not_be_used -o $output_xgb_model
python ml_3_evaluate_xgboost.py -d $input_file -f $feature_obs_not_be_used -o $output_xgb_model -p $plt_path -ocsv $output_inel_csv

