function [pairwise_table] = GeneratePairwiseSpreadsheet(T,sensor_feats,clin_vars,time_diff_range)

    % 'T' is the table of demographic data and feature data
    % 'sensor_feats' contains a cell array of feature names for modeling
    % 'clin_vars' contains a cell array of pertinent contextual variables: diagnosis, sex, age, etc
    % 'time_diff_range' contains the minimum and maximum length of time between pairwise comparisons
    
    %% let's only include the pertinent variables in the table
    table_subset = T(:,[clin_vars,sensor_feats]);
    
    %% shuffle order of initial table so that comparisons are randomized
    perm = randperm(size(table_subset,1));
    table_subset = table_subset(perm,:);
    
    %% create the cpairwise comparisons table
    pairwise_table = create_initial_pairwise_table(table_subset,clin_vars,sensor_feats);
    
    %% identify same participant and different participant comparisons
    pairwise_table.same_subj = pairwise_table{:,'subject'} == pairwise_table{:,'subject2'};
    pairwise_table = movevars(pairwise_table,'same_subj','Before',sensor_feats(1));
    
    %% remove control participant comparisons
    rows_to_remove = strcmp(pairwise_table.diagnosis,'Control') | strcmp(pairwise_table.diagnosis2,'Control');  
    pairwise_table = pairwise_table(~rows_to_remove,:);
    
    %% limit to only intrasubject comparisons
    pairwise_table = pairwise_table(pairwise_table.subject == pairwise_table.subject2,:);
    
    %% for same participant comparisons, remove rows where the time difference
    % between sessions is less than some thresh: abs(age2-age1) <=
    % time_diff_range
    rows_to_remove = pairwise_table.same_subj & (abs(pairwise_table.age2-pairwise_table.age) <= time_diff_range(1) | abs(pairwise_table.age2-pairwise_table.age) >= time_diff_range(2));  
    pairwise_table = pairwise_table(~rows_to_remove,:);
    
    %% for same participant comparisons, create classification target
    % (1 if age2-age1 > 0, 0 if age2-age1 < 0)
    pairwise_table.age_diff = pairwise_table.age2-pairwise_table.age;
    pairwise_table.target = nan(size(pairwise_table,1),1);
    pairwise_table{pairwise_table.same_subj & pairwise_table.age_diff > 0,'target'} = 0;
    pairwise_table{pairwise_table.same_subj & pairwise_table.age_diff < 0,'target'} = 1;
    pairwise_table = movevars(pairwise_table,'age_diff','Before',sensor_feats(1));
    pairwise_table = movevars(pairwise_table,'target','Before',sensor_feats(1));
    
    %% Additional comparisons can be removed here
    % for example, can filter out comparisons between sessions that were collected using different device firmwares


end

function [pairwise_table] = create_initial_pairwise_table(table_subset,clin_vars,sensor_feats)
    % repmat in one direction for n^2 rows
    temp = repmat(table_subset,size(table_subset,1),1);
    
    % create another table repmatted in the other direction for n^2 rows
    temp2 = temp; % just to prealocate
    table_vars = table_subset.Properties.VariableNames;
    for i = 1:length(table_vars)
        temp_mat = repmat(table_subset{:,table_vars(i)}',size(table_subset,1),1);
        temp2{:,table_vars(i)} = temp_mat(:);
    end
    
    % identify upper triangle of pairwise comparison matrix in order to remove
    % duplicates: N(N-1)/2 comparisons
    ones_vec = ones(size(table_subset,1),1);
    upper_tri_mat = repmat(ones_vec',size(table_subset,1),1);
    upper_tri_mat = triu(upper_tri_mat);
    upper_tri_vec = upper_tri_mat(:);
    
    temp2 = temp2(upper_tri_vec == 0,:);
    temp = temp(upper_tri_vec == 0,:);
    
    % rename clin vars for second repmatted table
    temp2 = renamevars(temp2,clin_vars,strcat(clin_vars,num2str(2)));
    
    % now we have a table of N(N-1)/2  comparisons
    pairwise_table = [temp(:,clin_vars),temp2];  
    
    % now lets create the feature differences for each comparison 2-1
    for i = 1:length(sensor_feats)
        pairwise_table{:,sensor_feats(i)} = temp2{:,sensor_feats(i)} - temp{:,sensor_feats(i)};
    end

end
