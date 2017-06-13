function [ ] = deconv_data_t4( RY )
%DECONV_DATA_T4 Summary of this function goes here
%   Detailed explanation goes here

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
% kernel = @lpsf_semi;     % Kernel used
kernel = @lpsf;
p_len = 5;                  % number of kernel parameters
p_eps = 0.06;
k_sparse = 3;               % number of peaks
niter = 200;      

%-Functions to generate observation and objective
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2;

%-Take a single line as truth
Y = RY(:,3)';

%- Initialize variables:
p_shape = [.05, 2, -3];
p_scale = [.05];

dY = Y(2:end) - Y(1:end-1);
ddY = dY(2:end) - dY(1:end-1);
dY_index = find(abs(dY)/norm(dY,2) > p_eps);
ddY_index = find(ddY >= 0);
Y_cand = Y;
Y_cand(dY_index) = 0;
Y_cand(ddY_index) = 0;
[~,I] = sort(Y_cand);
p_loc = I( (end - k_sparse + 1): end);
% p_loc(1) = 125;
% p_loc(3) = 130;
% p_loc(4) = 51;
% p_loc(7) = 59;


%-Use Good initializations: (for a specific sample...)
% p_task(1,:) = [5, 5, -2, 39, .005];
% p_task(2,:) = [5, 5, -2, 59, .005];
% p_task(1,:) = [.1, 2, -3, 39, 0.05];    % 'BEST FIT' LPSF
% p_task(2,:) = [.1, 2, -3, 59, 0.05];    % 'BEST FIT' LPSF

%-Use 'automatic' parameters:
for i = 1:k_sparse
    p_task(i,:) = [p_shape, p_loc(i), p_scale];
end
% p_task(2,4) = 64;

%-Gradient Step size + Jacobian differential
tp_scale = 0.05 * ones(1, size(p_task,2));
tp_scale(4) = 75;
tp_scale(5) = 0.05;

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
    parfor j = 1:size(p_task,1)
        for k = 1:p_len
            ek = zeros(size(p_task));
            ek(j,k) = 1;
            Yhat_eps = zeros(1,samples_num);

            Yhat_eps = Yhat + kernel(p_task(j,:) + dp.*ek(j,:) , range) ...
                        - kernel(p_task(j,:), range);
            Jp{j,k} = (Yhat_eps - Yhat) / dp(k);
            delta(j,k) = (tp_scale(k)/(norm(Jp{j,k},'fro')))*sum(sum(Jp{j,k}.*(Yhat - Y)));
        end
    end
    

    %Gradient step in p
    for j = 1:size(p_task,1)
        for k = 1:p_len
            p_task(j,k) = p_task(j,k) - delta(j,k);
        end
    end
   
    %Project onto correct subspace ASSUMING LPSF(_semi)
    p_task(:,1) = max(1e-3, p_task(:,1));
    p_task(:,2) = max(1e-3, p_task(:,2));
    p_task(:,3) = min(-1e-3, p_task(:,3));
    p_task(:,4) = max(1e-3, p_task(:,4));
    p_task(:,5) = max(1e-5, p_task(:,5));

    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    for j = 1:k_sparse
        disp(['p_test (',num2str(j),'): ', num2str(p_task(j,:))]);
    end
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

