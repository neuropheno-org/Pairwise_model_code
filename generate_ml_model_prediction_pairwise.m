function [model_name,Y_pred,Y_pred_3d] = generate_ml_model_prediction_pairwise(T,T_3d,data_source,sensor_feats,num_folds)

%% Input
% 'T' is the table with all data, including demographic and clinical data and sensor-based feature data; 
%       each row is data from a single session/sensor wear period
% 'T_3d' contains similar information as T, except data are now broken into the first and second half of the week with 
%       '_days1_3' or '_days4_6' as a suffix to the variable name
% 'sensor_feats' contains a cell array of feature names for modeling
% 'data_source' is the sensor location (e.g., wrist, ankle)
% 'num_folds' is the number of cross validation folds

%% Output
% 'model_name': the name of the model for saving purposes
% 'Y_pred': the model predictions
% 'Y_pred_3d': the model predictions for split week data

%% setup directory for saving the model information for each parfor loop
subdir_root = strcat(datestr(now,'YYYY_mm_DD_'),'pairwise_model_preds/'); % for where to save each parfor loop model info
if ~exist(subdir_root,'dir')
    mkdir(subdir_root)
end
model_suffix = '_ml_model';
sensor_feats_excluding_models = sensor_feats(~contains(sensor_feats,model_suffix)); % we don't want to include composite models in our pairwise training, just individual features

%% zscore each feature prior to model training
for i = 1:length(sensor_feats)
    T.(sensor_feats{i}) = nanzscore(T.(sensor_feats{i}));
    T_3d.(sensor_feats{i}) = nanzscore(T_3d.(sensor_feats{i}));
    T_3d.(strcat(sensor_feats{i},'_days1_3')) = nanzscore(T_3d.(strcat(sensor_feats{i},'_days1_3')));
    T_3d.(strcat(sensor_feats{i},'_days4_6')) = nanzscore(T_3d.(strcat(sensor_feats{i},'_days4_6')));
end

%% Create pairwise comparison table from T
% parameters for pairwise comparison table
clin_vars = {'subject','year','age','sex','diagnosis'};
time_diff_range = [0 Inf]; % [0 Inf] means that all comparisons are used; [0.25 0.5] would mean only comparisons separated in time by 3-6 months would be used
[pairwise_table] = GeneratePairwiseSpreadsheet(T,sensor_feats,clin_vars,time_diff_range);

%% Input and output
X = pairwise_table{:,sensor_feats_excluding_models};
Y = pairwise_table{:,'target'}; % this is the binary target created in GeneratePairwiseSpreadsheet

subjects1 = pairwise_table.subject;
subjects2 = pairwise_table.subject2;

%% Identify unique subjects to ensure data from each subject falls in single CV fold
unique_subjects = unique([subjects1;subjects2]);
if isempty(num_folds) %then LOSOCV
    num_folds = length(unique_subjects);
    fold_num = 1:num_folds;
else
    fold_num = repmat(1:num_folds,1,ceil(length(unique_subjects)/num_folds));
    fold_num = fold_num(1:length(unique_subjects));
end
test_subject_fold_pairing = [unique_subjects,fold_num']; % to keep track of which test subject is part of which fold

%% Initialize variables
Y_pred = nan(size(Y));
model_name = strcat(data_source,'_intra_subject_pairwise_ml_model');
selected_feats_matrix = nan(length(sensor_feats_excluding_models),num_folds); % to keep track of which features are included in each model trained (one per fold)

%% Initiate Outer CV Loop
parfor i = 1:num_folds
        
    sprintf('############ CV LOOP %d ############',i)

    % curr_subjects contains our test subjects
    curr_subjects = unique_subjects(fold_num == i)

    % Train and test split for CV outer loop
    test_subject_indices = boolean(zeros(size(subjects1)));
    for j = 1:length(curr_subjects)
        test_subject_indices = test_subject_indices | (subjects1 == curr_subjects(j) | subjects2 == curr_subjects(j));
    end
    train_subject_indices = ~test_subject_indices & ~isnan(sum(X,2));

    X_train = X(train_subject_indices,:);
    Y_train = Y(train_subject_indices);

    %% Model training
    [B,FitInfo] = lassoglm(X_train,Y_train,'binomial','CV',3);
    idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
    B0 = FitInfo.Intercept(idxLambdaMinDeviance);
    coef = [B0; B(:,idxLambdaMinDeviance)];
    curr_selected_feats = B(:,idxLambdaMinDeviance);

    %% Model prediction
    % to parallelize each fold in parfor loop, we have to save each fold's data to disk
    % we also save each test subject's predictions separately for transparency
    for j = 1:length(curr_subjects)
        single_test_subject_indices = subjects1 == curr_subjects(j) | subjects2 == curr_subjects(j);
        X_test_single_subject = X(single_test_subject_indices,:);
        Y_test_single_subject = Y(single_test_subject_indices);
        curr_Y_pred = glmval(coef,X_test_single_subject,'logit');
        curr_Y_true = Y_test_single_subject;
        num_predictions = length(Y);            
        parsave_generate_ml_model_preds(subdir_root,i,curr_subjects(j),curr_Y_pred,curr_Y_true,curr_selected_feats,num_predictions,single_test_subject_indices,unique_subjects,data_source)
    end
        
end

%% Training and predictions are complete at this stage. Code below is for evaluating performance of the model.
%% in order to evaluate model performance, one can loop through the subject files saved in the 'parsave_generate_ml_model_preds' step above
% for example:
all_fn = dir(subdir_root);
folders = {all_fn.folder};
fn = {all_fn.name};
fn = fn(contains(fn,'.mat') & contains(fn,data_source));

temp_data = load(strcat(folders{1},'/',fn{1}));

Y_pred_binary = nan(size(temp_data.test_subject_indices));
Y_true_binary = Y_pred_binary;
Y_pred_binary_subject = nan(length(fn),1);
selected_feats_matrix_indiv_subject = nan(length(sensor_feats_excluding_models),length(fn));
selected_feats_matrix = nan(length(sensor_feats_excluding_models),num_folds);

for i = 1:length(fn)
    curr_full_fn = load(strcat(folders{1},'/',fn{i}));
    Y_pred_binary(curr_full_fn.test_subject_indices) = curr_full_fn.curr_Y_pred;
    Y_true_binary(curr_full_fn.test_subject_indices) = curr_full_fn.curr_Y_true;
    Y_pred_binary_subject(i) = curr_full_fn.curr_subject;
    selected_feats_matrix_indiv_subject(:,i) = curr_full_fn.curr_selected_feats;
    selected_feats_matrix(:,curr_full_fn.fold_num) = curr_full_fn.curr_selected_feats;
end

binarized_Y_pred = Y_pred_binary > 0.5;
disp('accuracy is this:')
sum(binarized_Y_pred == Y_true_binary)/length(binarized_Y_pred)




