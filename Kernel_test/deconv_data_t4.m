function [ ] = deconv_data_t4( RY )
%DECONV_DATA_T4 Summary of this function goes here
%   Detailed explanation goes here

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
kernel = @lpsf_semi;     % Kernel used
p_len = 4;
% kernel = @lpsf;
% p_len = 5;
k_sparse = 2;
niter = 200;      

%-Functions to generate observation and objective
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2;

%- Initialize variables:
p_task = [];

%-Use Good initializations:
p_task(1,:) = [5, -2, 39, .005];
p_task(2,:) = [5, -2, 59, .005];
% p_task(1,:) = [.1, 2, -3, 39, 0.05];    % 'BEST FIT' LPSF
% p_task(2,:) = [.1, 2, -3, 59, 0.05];    % 'BEST FIT' LPSF

%-Take a single line as truth
Y = RY(:,2)';

%-Gradient Step size initialization
tp = 0.05/(norm(Y,'fro').^2) * ones(1, p_len);
% tp(5) = 0.001;      % Super hacky... otherwise the scale blows up
tp(4) = 0.001;
dp = 0.01 * ones(1, size(p_task,2)); 

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
        for k = 1:p_len
            ek = zeros(size(p_task));
            ek(j,k) = 1;
            Yhat_eps = zeros(1,samples_num);
            for l = 1:k_sparse
                Yhat_eps = Yhat_eps + kernel(p_task(l,:) + dp.*ek(l,:) , range);
            end
            Jp{j,k} = (Yhat_eps - Yhat) / dp(k);
        end
    end
    

    %Gradient step in p
    for j = 1:size(p_task,1)
        for k = 1:p_len
            change = tp(k)*sum(sum(Jp{j,k}.*(Yhat - Y)));
            p_task(j,k) = p_task(j,k) - change;
        end
    end

    %Project onto correct subspace ASSUMING LPSF_SEMI
    p_task(:,1) = max(1e-3, p_task(:,1));      
    p_task(:,2) = min(-1e-3, p_task(:,2));     
    p_task(:,3) = max(1e-3, p_task(:,3));
    p_task(:,4) = max(1e-5, p_task(:,4));    
   
    %Project onto correct subspace ASSUMING LPSF
    %p_task(:,1) = max(1e-3, p_task(:,1));
    %p_task(:,2) = max(1e-3, p_task(:,2));
    %p_task(:,3) = min(-1e-3, p_task(:,3));
    %p_task(:,4) = max(1e-3, p_task(:,4));
    %p_task(:,5) = max(1e-5, p_task(:,5));

    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    disp(['p_test (1): ', num2str(p_task(1,:))]);
    disp(['p_test (2): ', num2str(p_task(2,:))]); 
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

