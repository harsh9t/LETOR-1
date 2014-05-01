%Please run this file to see the results. It calls train.m for each value
%of M.
data_set = load('project1_data.mat');
raw_data = data_set.raw_data;

%range of values of M, from 1 to 30
degree_limit = 30;

%regularization parameter, lambda - you can set it manually to determine
%for what value is the model gives best performance
lambda = 0.001;

%3 columns store [training, validation, testing] errors respectively
Errors = zeros([degree_limit 3]);

%Additional logging for errors calucated of each degree, comment if neeed
disp('        Error_training            Error_validation          Error_testing');

%Training the model for each value of M, storing the errors in Errors
for d = 1:degree_limit
    [Errors(d,1),Errors(d,2),Errors(d,3)] = train(raw_data, d, lambda);    
    %Additional logging for errors calucated of each degree, comment if needed
    disp(Errors(d,:));
end

training_errors = Errors(:,1);
validation_errors = Errors(:,2);
testing_errors = Errors(:,3);

[rms_lr,M] = min(testing_errors(:));

%Plot of Erms vs. M
%Validation error is commented intentionally, please uncomment to view it
plot(1:degree_limit, training_errors, '-b.');
hold on;
plot(1:degree_limit, testing_errors, '-r.');
%plot(1:degree_limit, validation_errors, '-g.');
set(gca, 'XTick', 1:degree_limit);
title('Plot of Error vs. M');
xlabel('M');
ylabel('Error');
legend('Error_t_r_a_i_n_i_n_g','Error_t_e_s_t_i_n_g');
%legend('Error_t_r_a_i_n_i_n_g','Error_t_e_s_t_i_n_g','Error_v_a_l_i_d_a_t_i_o_n');
%text(1:degree_limit,validation_errors,num2str(validation_errors),'HorizontalAlignment','left');
text(1:degree_limit,training_errors,num2str(training_errors),'HorizontalAlignment','left');
text(1:degree_limit,testing_errors,num2str(testing_errors),'HorizontalAlignment','left');

%Priting final output
fprintf('the model complexity M for the linear regression model is %d \n', M);
fprintf('the regularization parameters lambda for the linear regression model is %f \n', lambda);
fprintf('the root mean square error for the linear regression model is %f \n', rms_lr);