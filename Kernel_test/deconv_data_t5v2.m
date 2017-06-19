function [ ] = deconv_data_t5( RY )
%DECONV_DATA_T4 Summary of this function goes here
%   Detailed explanation goes here

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
% kernel = @lpsf_semi;     % Kernel used
kernel = @lpsf;
p_len = 4;                  % number of kernel parameters
niter = 150;      
LAMBDA_SCALE = 0.5;



%-Take a single line as truth
Y = RY(:,2);

%- Initialize variables:
x = 0.05 * ones(length(range), 1);
p_shape = [.05, 2, -3];

%-Use 'automatic' parameters:
for i = 1:samples_num
    p_task(i,:) = [p_shape, range(i)];
end

%-Gradient Step size + Jacobian differential
tp_scale = 1 * ones(1, size(p_task,2));
tp_scale(4) = 0;

dp = 0.01 * ones(1, size(p_task,2)); 

tx = 0.001;
lambda = LAMBDA_SCALE*max(max(abs(Y)));

%-Functions to generate observation and objective
objective = @(Yhat, Y, x, lambda) 0.5*norm(Yhat - Y,2)^2 + lambda * norm(x,1);

%% (TASK 4) Estimate p %%
% For this task the objective function used will be:
% min 0.5 * || D(p) * x - y ||_2 ^2 + lambda ||x||_1
%       - d: kernel function
Yhat = zeros(1,samples_num);
Yhat_mat = zeros(samples_num, samples_num);
for j = 1:samples_num
    Yhat_mat(j,:) = kernel(p_task(j,:), range);
end
Yhat = Yhat_mat' * x;

error = zeros(1,niter);
lambda_hist = zeros(1,niter);
tx_hist = zeros(1,niter);
for i = 1:niter

    %Update Parameters
    lambda = LAMBDA_SCALE*max(max(abs(Yhat_mat * Y)));
    
    if (mod(i,3) == 0)
        Z = randn(samples_num,1);
        Z = Z/norm(Z,'fro');
        for iiter = 1:20
            Z = (Yhat_mat * (Yhat_mat' * Z));
            Z = Z/norm(Z,'fro'); 
        end
        Lip = norm(Yhat_mat * (Yhat_mat' * Z),'fro');
        tx = 0.5/Lip;
    end
    lambda_hist (i) = lambda;
    tx_hist (i) = tx;
    e = objective(Yhat, Y, x, lambda);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(size(p_task));
    parfor j = 1:size(p_task,1)
        for k = 1:p_len
            ek = zeros(size(p_task));
            ek(j,k) = 1;

            Yhat_mat_eps = Yhat_mat;
            Yhat_mat_eps(j,:) = kernel(p_task(j,:) + dp.*ek(j,:) , range);
            Yhat_eps = Yhat_mat_eps' * x;

            Jp{j,k} = (Yhat_eps - Yhat) / dp(k);
            delta(j,k) = (tp_scale(k)/(norm(Jp{j,k},'fro')))*sum(sum(x(j) * Jp{j,k}.*(Yhat - Y)));
        end
    end
    

    %Gradient step in p
    for j = 1:size(p_task,1)
        for k = 1:p_len
            p_task(j,k) = p_task(j,k) - delta(j,k);
        end
    end
   
   %Gradient step in x
    for j = 1:samples_num
        Yhat_mat(j,:) = kernel(p_task(j,:), range);
    end
    Yhat = Yhat_mat' * x;
    x = x - tx * Yhat_mat * (Yhat - Y);
    x = sign(x).*max(abs(x - lambda*tx), 0);

    %Project onto correct subspace ASSUMING LPSF(_semi)
    p_task(:,1) = max(1e-12, p_task(:,1));
    p_task(:,2) = max(1e-12, p_task(:,2));
    p_task(:,3) = min(-1e-12, p_task(:,3));
    p_task(:,4) = max(1e-12, p_task(:,4));
    x = max(1e-12, x);


    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    for j = 1:size(p_task,1)
        if(x(j) > 1e-2)
            disp(['p_test (',num2str(j),'): ', num2str(p_task(j,:)),'   ', num2str(x(j))]);
        end
    end
end

%% Visualization %%
figure(1); clf;

subplot(4,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
title('Truth vs. Learned');
hold off;

subplot(4,1,2);
hold on;
plot(lambda_hist);
plot(tx_hist);
hold off;
xlabel('Number of Iterations');
ylabel('Step Value');
title('Step Scale');

subplot(4,1,3);
stem(x);
xlabel('Magnitude');
ylabel('Location');
title('Learned Activation Map');

subplot(4,1,4);
plot(error(2:end));
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

end

