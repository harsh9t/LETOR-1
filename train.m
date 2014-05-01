function [etr,evl,ets] = train(raw_data, degree, regularizationConst)

format longE
M=degree;
lambda = regularizationConst;

raw_rownum = length(raw_data(:,1));




training_rownum = int32(0.4 * raw_rownum); 
validation_rownum = int32(0.1 * raw_rownum);
testing_rownum = int32(0.5 * raw_rownum);



%Calculating the training set, validtion set and the testing set
training_data = raw_data(1:training_rownum,:);
validation_data = raw_data(training_rownum+1:training_rownum + validation_rownum,:);
testing_data = raw_data(training_rownum + validation_rownum + 1: training_rownum + validation_rownum + testing_rownum,:);



training_target = training_data(:,1);
training_set = training_data(:,2:end);

%CALUCALTING BASIS FUNCTIONS - MEANS, VARIANCES FOR EACH
dimensions = length(training_set(1,:));
partition_size = floor(dimensions/M);
remainder_cols = rem(dimensions,partition_size);

means = zeros([1 (partition_size + remainder_cols)]);
variances = zeros([1 (partition_size + remainder_cols)]);
basis_params = zeros([M,2]);

for i = 1:M
numcols = partition_size;
if i == M
numcols = partition_size + remainder_cols;
end
for j = 1:numcols
means(j) = mean(training_set(:,(i-1)*partition_size+j));
variances(j) = var(training_set(:,(i-1)*partition_size+j));
end
basis_params(i,1) = mean(means);
basis_params(i,2) = mean(variances) + 0.00001;
end






%TRAINING PHASE
num_trainingsamples = length(training_set(:,1));

%Calculation of the design matrix
design_matrix = zeros([num_trainingsamples, (dimensions*M+1)]);
design_matrix(:,1) = 1;
for i = 1:M
start_index = (i-1)*dimensions+2;
end_index = start_index + dimensions - 1;
for r = 1:num_trainingsamples
for c = start_index:end_index
design_matrix(r,c) = exp(-(((training_set(r,c-(1+(i-1)*dimensions))- basis_params(i,1))^2)/(2*(basis_params(i,2)^2))));
end
end
end

%Calculation of training weights
tr_design_matrix = transpose(design_matrix);
weights = (tr_design_matrix*design_matrix + lambda*eye(length(design_matrix(1,:))))\tr_design_matrix*training_target;

%Calculation fo root mean square error for training
Error_training = sqrt(((transpose(design_matrix*weights - training_target))*(design_matrix*weights - training_target))/num_trainingsamples);
etr = Error_training;






%VALIDATION PHASE
validation_target = validation_data(:,1);
validation_set = validation_data(:,2:end);
num_validationsamples = length(validation_set(:,1));

%Calculation of design matrix
validation_design_matrix = zeros([num_validationsamples, (dimensions*M+1)]);
validation_design_matrix(:,1) = 1;

for i = 1:M
start_index = (i-1)*dimensions+2;
end_index = start_index + dimensions - 1;
for r = 1:num_validationsamples
for c = start_index:end_index
validation_design_matrix(r,c) = exp(-(((validation_set(r,c-(1+(i-1)*dimensions))- basis_params(i,1))^2)/(2*(basis_params(i,2)^2))));
end
end
end

%Calculation of validation weights
tr_validation_design_matrix = transpose(validation_design_matrix);
validation_weights = (tr_validation_design_matrix*validation_design_matrix + lambda*eye(length(validation_design_matrix(1,:))))\tr_validation_design_matrix*validation_target;

%Calculation of root mean square error for validation
Error_validation = sqrt(((transpose(validation_design_matrix*validation_weights - validation_target))*(validation_design_matrix*validation_weights - validation_target))/num_validationsamples);
evl = Error_validation;






%TESTING PHASE
testing_target = testing_data(:,1);
testing_set = testing_data(:,2:end);
num_testingsamples = length(testing_set(:,1));

%Calculation of design matrix
testing_design_matrix = zeros([num_testingsamples, (dimensions*M+1)]);
testing_design_matrix(:,1) = 1;

for i = 1:M
start_index = (i-1)*dimensions+2;
end_index = start_index + dimensions - 1;
for r = 1:num_testingsamples
for c = start_index:end_index
testing_design_matrix(r,c) = exp(-(((testing_set(r,c-(1+(i-1)*dimensions))- basis_params(i,1))^2)/(2*(basis_params(i,2)^2))));
end
end
end

%Calculation of testing weights
tr_testing_design_matrix = transpose(testing_design_matrix);
testing_weights = (tr_testing_design_matrix*testing_design_matrix + lambda*eye(length(testing_design_matrix(1,:))))\tr_testing_design_matrix*testing_target;

%Calculation of root mean square error for validation
Error_testing = sqrt(((transpose(testing_design_matrix*testing_weights - testing_target))*(testing_design_matrix*testing_weights - testing_target))/num_testingsamples);
ets = Error_testing;

end

