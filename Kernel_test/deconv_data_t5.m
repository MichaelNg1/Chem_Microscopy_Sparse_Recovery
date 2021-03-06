function [ ] = deconv_data_t5( RY )
%DECONV_DATA_T5 Attempts to fit the observation by placing a kernel per pixel.
% Here we use all the kernel parameters (i.e. shape, location, magnitude)
% The objective minimizes reconstruction and uses sparsity regularization.

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
% kernel = @lpsf_semi;     % Kernel used
kernel = @lpsf;
p_len = 5;                  % number of kernel parameters
p_eps = 0.06;
niter = 200;      
LAMBDA_SCALE = 0.5;



%-Take a single line as truth
Y = RY(:,1)';

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
scale_step = zeros(1,niter);
for i = 1:niter
    Yhat = zeros(1,samples_num);
    Yhat_mat = zeros(samples_num, samples_num);
    for j = 1:samples_num
        Yhat_mat(j,:) = kernel(p_task(j,:), range);
        Yhat = Yhat + Yhat_mat(j,:);
    end

    %Update Parameters
    lambda = LAMBDA_SCALE*max(max(abs(Yhat)));
    
    if (mod(i,3) == 0)
        Z = randn(samples_num,1);
        Z = Z/norm(Z,'fro');
        for iiter = 1:20
            Z = (Yhat_mat * (Yhat_mat' * Z));
            Z = Z/norm(Z,'fro'); 
        end
        Lip = 2*norm(Yhat_mat * (Yhat_mat' * Z),'fro');
        tx = 0.0002/Lip;
        tp_scale(5) = tx;
        
    end
    scale_step(i) = lambda;
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
            if k == 5
                p_task(j,k) = p_task(j,k) - tp_scale(k) * sum(sum(Jp{j,k}.*(Yhat - Y)));
                p_task(j,k) = sign(p_task(j,k)).*max(abs(p_task(j,k))-lambda*tx,0);
            else
                p_task(j,k) = p_task(j,k) - delta(j,k);
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

subplot(4,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
title('Truth vs. Learned');
hold off;

subplot(4,1,2);
plot(scale_step);
xlabel('Number of Iterations');
ylabel('Step Value');
title('Step Scale');

subplot(4,1,3);
stem(p_task(:,5));
xlabel('Magnitude');
ylabel('Location');
title('Learned Activation Map');

subplot(4,1,4);
plot(error(2:end));
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

end

