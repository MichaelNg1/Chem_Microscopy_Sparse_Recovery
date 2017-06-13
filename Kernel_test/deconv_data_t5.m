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
niter = 150;      
LAMBDA_SCALE = 0.5;



%-Take a single line as truth
Y = RY(:,5)';

%- Initialize variables:
p_shape = [.05, 2, -3];
p_scale = [.05];

%-Use 'automatic' parameters:
for i = 1:samples_num
    p_task(i,:) = [p_shape, range(i), p_scale];
end
% p_task(2,4) = 64;

%-Gradient Step size + Jacobian differential
tp_scale = 0.1 * ones(1, size(p_task,2));
tp_scale(4) = 0;
tp_scale(5) = 0.1;

dp = 0.01 * ones(1, size(p_task,2)); 

tx = 0.05;
lambda = LAMBDA_SCALE*max(max(abs(Y)));

%-Functions to generate observation and objective
objective = @(Yhat, Y, p) 0.5*norm(Yhat - Y,2)^2 + lambda * norm(p(:,5),1);

%% (TASK 4) Estimate p %%
% For this task the objective function used will be:
% min 0.5 * || D(p) * x - y ||_2 ^2 + lambda ||x||_1
%       - d: kernel function
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    Yhat = zeros(1,samples_num);
    for j = 1:samples_num
        Yhat = Yhat + kernel(p_task(j,:), range);
    end

    %Update Parameters
    lambda = LAMBDA_SCALE*max(max(abs(Yhat)));
    

    e = objective(Yhat, Y, p_task);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(size(p_task));
    parfor j = 1:size(p_task,1)
        for k = 1:p_len
            ek = zeros(size(p_task));
            ek(j,k) = 1;

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
            if k == 5
                p_task(j,k) = sign(p_task(j,k)).*max(abs(p_task(j,k))-lambda*tx,0);
            end
        end
    end
   
    %Project onto correct subspace ASSUMING LPSF(_semi)
    p_task(:,1) = max(1e-3, p_task(:,1));
    p_task(:,2) = max(1e-3, p_task(:,2));
    p_task(:,3) = min(-1e-3, p_task(:,3));
    p_task(:,4) = max(1e-3, p_task(:,4));
    p_task(:,5) = max(1e-12, p_task(:,5));

    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    for j = 1:size(p_task,1)
        if(p_task(j,5) > 1e-5)
            disp(['p_test (',num2str(j),'): ', num2str(p_task(j,:))]);
        end
    end
end

%% Visualization %%
figure(1); clf;

subplot(3,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
title('Truth vs. Learned');
hold off;

subplot(3,1,2);
plot(error(2:end));
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

subplot(3,1,3);
stem(p_task(:,5));
xlabel('Magnitude');
ylabel('Location');
title('Learned Activation Map');

end

