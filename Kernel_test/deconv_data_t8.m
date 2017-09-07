function [ ] = deconv_data_t8(RY, LD, D, params_init, dim,lambda)
%DECONV_DATA_T7 Attempts to fit the observation with a finite number of kernels.
% The locations are chosen via 1st/2nd gradient information without prior knowledge on number of peaks.
% Here we use only shape parameters and a vector of scalars

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
% kernel = @lpsf_semi;     % Kernel used
kernel = @lpsf;
SELECT_LINE = 4;
SELECT_DATA = 6;
SAVE_DATA = 0;
p_len = 3;                  % number of kernel parameters
p_eps = 0.06;
niter = 200;      

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
p_task = zeros(k_sparse, p_len);
for i = 1:k_sparse
    p_task(i,:) = [p_shape];
end

% Use the parameter per pixel: 
D_kernel = zeros( samples_num );
for i = 1:k_sparse
    location = range( p_loc(i) );
    D_kernel(:, location ) =  kernel([p_task(i,:), location], range);
end

%- Initialize Activation Map
X = RY_INIT( :, SELECT_LINE );
Xorg = X;

%-Gradient Step size + Jacobian differential
tp_scale = 1.5 * ones(1, size(p_task,2));
tx = 0.5;
dp = 0.01 * ones(1, size(p_task,2)); 

Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    % Get Yhat and the new kernel matrix
    D_kernel = zeros( samples_num );
    for j = 1:k_sparse
        location = range( p_loc(j) );
        D_kernel(:, location ) =  kernel([p_task(j,:), location], range);
    end
    Yhat = D_kernel * X;

    e = objective(Yhat, Y');
    error(i) = e;
    
    %Gradient step in X
    g_step = D_kernel' * (Yhat - Y');
    X = X - tx * g_step;

    %Estimate Jacobian
    Jp = cell(size(p_task));
    parfor j = 1:size(p_task,1)
        for k = 1:p_len
            ek = zeros(size(p_task));
            ek(j,k) = 1;

            location = range( p_loc(j) );
            D_kernel_eps = D_kernel;
            D_kernel_eps(:, location) = ...
                kernel([p_task(j,:) + dp.*ek(j,:), location], range);

            Yhat_eps = D_kernel_eps * X;
            Jp{j,k} = (Yhat_eps - Yhat) / dp(k);
            delta(j,k) = (tp_scale(k)/(norm(Jp{j,k},'fro')))* ...
                sum(sum(X(location) * Jp{j,k}.*(Yhat - Y')));
        end
    end

    %Gradient step in p
    for j = 1:size(p_task,1)
        for k = 1:p_len
            p_task(j,k) = p_task(j,k) - delta(j,k);
        end
    end
   
    %Project onto correct subspace ASSUMING LPSF(_semi)
    p_task(:,1) = max(1e-4, p_task(:,1));
    p_task(:,2) = max(1e-4, p_task(:,2));
    p_task(:,3) = min(-1e-4, p_task(:,3));

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
stem(range, X);
xlabel('range');
title('Activation Map');

if( SAVE_DATA )
    filename = ['DATA' num2str(SELECT_DATA) '_LINE' num2str(SELECT_LINE) '_peaks' num2str(k_sparse) '_' date];
    saveas(gcf, [filename '.png']);
    save([filename '.mat'], 'p_task');
end

end

