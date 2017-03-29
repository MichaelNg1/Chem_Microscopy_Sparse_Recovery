%% [Description] This script attempts to find the kernel parameters given the kernel
%shape and the activation maps of multiple lines.

clear all;
%-Add the path for original lpsf function
addpath('..\Chem_microscopy_code');

%% Constants %%
samples_num = 256;  % Number of samples
kernel = @lpsf;     % Kernel used
niter = 250;         % Number of iterations
k_num = 7;          % Number of "spikes"

%% Generate Artificial Test Data %%
%-Generate artificial activation map
xi = ceil(samples_num * rand(1,k_num));
x_mag = ones(1,numel(xi));      
range = 1:samples_num;          % assume unit distance
activation_map = zeros(1, samples_num);
activation_map(xi) = 1;

%-Assign truth parameters for the kernel
p = [1, 1.5, -2];

%-Integration factors
dp = 0.01 * ones(1,3);

%-Functions to generate observation and objective
gen_y = @(kernel, range, map, p) conv(kernel(p, range), map, 'same');
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2;

%-Generate observation
Y = gen_y(kernel, range, activation_map, p);

%-Construct adjoint 
CDf = convmtx(kernel(p, range), length(range));
dl = floor(length(range)/2) + 1;
dr = ceil(length(range)/2);
conv_resize = dl:(2*length(range) - dr);

Y_mat = activation_map*CDf;
Y_mat_resize = Y_mat(1, conv_resize);      %Should be identitcal

obj_adj_long = @(map, range, p) (map*convmtx(kernel(p, range), length(range)) - Y_mat) ...
*convmtx(kernel(p, range), length(range))';
% obj_adj = @(map) obj_adj_long(map)(1,conv_resize);

%-Gradient Step size initialization
tp = 0.7/(norm(Y,'fro').^2); % Think about this
tx = tp; % Change this to Lipschitz

%% (TASK 1) Estimate p given xi %%
% For this task the objective function used will be:
% min || sum_(xi) d(x - xi)(p) - y ||_2
%       - d: kernel function

%- Initialize variables:
p_task = zeros(1,length(p)) + 1;
x_task = zeros(1,length(activation_map)) + 1;
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    Yhat = gen_y(kernel, range, x_task, p_task);
    e = objective(Yhat, Y);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(1,length(p_task));
    for k = 1:length(p_task)
        ek = zeros(size(p_task));
        ek(k) = 1;
        Yhat_eps = gen_y(kernel, range, x_task, p_task + dp.*ek);
        Jp{k} = (Yhat_eps - Yhat) / dp(k);
    end
    
    %Gradient step in p
    for k = 1:length(p_task)
        p_task(k) = p_task(k) - tp*sum(sum(Jp{k}.*(Yhat - Y)));
    end
    
    %Gradient step in x
    grad_f = obj_adj_long(x_task, range, p_task);
    x_task = x_task - tx*grad_f;

    %Project onto correct subspace
    p_task(1) = max(1e-10, p_task(1));      %assuming lpsf
    p_task(2) = max(1e-10, p_task(2));      %assuming lpsf
    p_task(3) = min(-1e-10, p_task(3));     %assuming lpsf
    
    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    disp(['p_test: ', num2str(p_task)]);
    disp(['p_real: ', num2str(p)])
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
plot(error);
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

subplot(4,1,3);
plot(kernel(p_task,range));
title('Kernel Shape');

subplot(4,1,4);
hold on;
plot(activation_map);
plot(x_task);
legend('Truth', 'Learned');
hold off;
title('Learned Activation Map');


