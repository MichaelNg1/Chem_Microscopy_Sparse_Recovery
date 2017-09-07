function [ ] = deconv_data_t7(RY, LD, D, params_init, dim,lambda)
%DECONV_DATA_T4 Attempts to fit the observation with a finite number of kernels.
% The locations are chosen via 1st/2nd gradient information without prior knowledge on number of peaks.
% Here we use all the kernel parameters (i.e. shape, location, magnitude)
% The objective minimizes reconstruction and uses sparsity regularization.

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
% kernel = @lpsf_semi;     % Kernel used
kernel = @lpsf;
SELECT_LINE = 2;
SELECT_DATA = 7;
p_len = 5;                  % number of kernel parameters
p_eps = 0.06;
niter = 500;      

%-Functions to generate observation and objective
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2;

%- Initialize variables:
p_shape = [.1, 2, -1.5];
p_scale = [.05];

theta = params_init.Rotation/180*pi; 
n = dim.n;
offset = dim.Offset;
Cy = dim.Cy;

%- Flip the signal if the direction is negative
direction = dim.Direction;
if( direction(SELECT_LINE) == -1 )
    RY(:, SELECT_LINE) = flip( RY(:, SELECT_LINE) );
end

%-Take a single line as truth
Y = RY(:,SELECT_LINE)';

[~, RY_INIT] = initialize_peaks( RY, theta, n, offset, Cy );
RY_MEANS = 1.75 * mean(RY_INIT);

%- Threshold out all the non-peaks
RY_INIT(RY_INIT(:,SELECT_LINE) < RY_MEANS( SELECT_LINE ), SELECT_LINE ) = 0;
p_loc = find(RY_INIT(:, SELECT_LINE));
k_sparse = length( p_loc );

%-Use 'automatic' parameters:
for i = 1:k_sparse
    p_task(i,:) = [p_shape, range( p_loc(i) ), p_scale];
end

%-Gradient Step size + Jacobian differential
tp_scale = 0.1 * ones(1, size(p_task,2));
tp_scale(4) = 0;
tp_scale(5) = 0.1;

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


filename = ['DATA' num2str(SELECT_DATA) '_LINE' num2str(SELECT_LINE) '_peaks' num2str(k_sparse) '_' date];
saveas(gcf, [filename '.png']);
save([filename '.mat'], 'p_task');

end

