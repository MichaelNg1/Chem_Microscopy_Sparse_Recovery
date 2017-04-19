function [ ] = deconv_data_t4( RY )
%DECONV_DATA_T4 Summary of this function goes here
%   Detailed explanation goes here

%-Add the path for original lpsf function
addpath('..\Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
kernel = @lpsf;     % Kernel used
p_len = 5;
k_sparse = 2;
niter = 50;      
% Number of iterations
LAMBDA_SCALE = 0.5;

%-Integration factors
dp = 0.01 * ones(1,p_len);

%-Functions to generate observation and objective
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2 + sum(abs(Yhat));

%- Initialize variables:
p_task = zeros(k_sparse ,p_len) + 1;

%-Take a single line as truth
Y = RY(:,2)';

%-Gradient Step size initialization
tp = 0.5/(norm(Y,'fro').^2);

%% (TASK 4) Estimate p %%
% For this task the objective function used will be:
% min 0.5 * || D(p) * x - y ||_2 ^2 + lambda ||x||_1
%       - d: kernel function
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    Yhat = zeros(1,samples_num);
    for j = 1:k_sparse
        Yhat = Yhat + kernel(p_task(j,:), range);
    end

    e = objective(Yhat, Y);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(size(p_task));
    for j = 1:size(p_task,1)
        for k = 1:size(p_task,2)
            ek = zeros(size(p_task));
            ek(j,k) = 1;
            Yhat_eps = zeros(1,samples_num);
            for l = 1:k_sparse
                Yhat_eps = Yhat_eps + kernel(p_task(j,:) + dp.*ek(l,:) , range);
            end
            Jp{j,k} = (Yhat_eps - Yhat) / dp(k);
        end
    end
    
    %Gradient step in p
    for j = 1:size(p_task,1)
        for k = 1:size(p_task,2)
            p_task(j,k) = p_task(j,k) - tp*sum(sum(Jp{j,k}.*(Yhat - Y)));
        end
    end

    %Project onto correct subspace
    p_task(1) = max(1e-1, p_task(1));      %assuming lpsf
    p_task(2) = max(1e-1, p_task(2));      %assuming lpsf
    p_task(3) = min(-1e-1, p_task(3));     %assuming lpsf
    
    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    disp(['p_test (1): ', num2str(p_task(1,:))]);
    
end

%% Visualization %%
figure(1); clf;

subplot(2,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
title('Truth vs. Learned');
hold off;

subplot(2,1,2);
plot(error);
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

end

