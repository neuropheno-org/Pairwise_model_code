function [] = parsave_generate_ml_model_preds(subdir_root,fold_num,curr_subject,curr_Y_pred,curr_Y_true,curr_selected_feats,num_predictions,test_subject_indices,unique_subjects,data_source)
    % this function simply saves key data from each training fold to file
    % to enable parallelization of folds during training
    save_fn = strcat(subdir_root,num2str(curr_subject),'_pairwise_model_',data_source,'.mat');
    save(save_fn,'fold_num','curr_subject','curr_Y_pred','curr_Y_true','curr_selected_feats','num_predictions','test_subject_indices','unique_subjects','data_source');

end