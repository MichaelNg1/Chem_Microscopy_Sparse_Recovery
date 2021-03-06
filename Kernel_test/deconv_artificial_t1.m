%% [Description] This script attempts to find the kernel parameters given the kernel
%shape and the activation map. This script looks at single lines (no line integration)

clear all;
%-Add the path for original lpsf function
addpath('..\Chem_microscopy_code');

%% Constants %%
samples_num = 128;  % Number of samples
kernel = @lpsf;     % Kernel used
niter = 250;         % Number of iterations
k_num = 5;          % Number of "spikes"

%% Generate Artificial Test Data %%
%-Generate artificial activation map
xi = ceil(samples_num * rand(1,k_num));
x_mag = ones(1,numel(xi));      
range = 1:samples_num;          % assume unit distance
activation_map = zeros(1, samples_num);
activation_map(xi) = 1;

%-Assign truth parameters for the kernel
p = [2, 1.5, -1];

%-Integration factors
dp = 0.01 * ones(1,3);

%-Functions to generate observation and objective
gen_y = @(kernel, x, map, p) conv(kernel(p,x), map, 'same');
objective = @(Yhat, Y) norm(Yhat - Y,2)^2;

%-Generate observation
Y = gen_y(kernel, range, activation_map, p);

%-Gradient Step size initialization
tp = 0.7/(norm(Y,'fro').^2); % Think about this

%% (TASK 1) Estimate p given xi %%
% For this task the objective function used will be:
% min || sum_(xi) d(x - xi)(p) - y ||_2
%       - d: kernel function

%- Initialize variables:
% p_task = zeros(1,length(p)) + 0.5;
p_task = [3 10 3];
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    Yhat = gen_y(kernel, range, activation_map, p_task);
    e = objective(Yhat, Y);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(1,length(p_task));
    for k = 1:length(p_task)
        ek = zeros(size(p_task));
        ek(k) = 1;
        Yhat_eps = gen_y(kernel, range, activation_map, p_task + dp.*ek);
        Jp{k} = (Yhat_eps - Yhat) / dp(k);
    end
    
    %Gradient step
    for k = 1:length(p_task)
        p_task(k) = p_task(k) - tp*sum(sum(Jp{k}.*(Yhat - Y)));
    end
    
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

subplot(3,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
title('Truth vs. Learned');
hold off;

subplot(3,1,2);
plot(error);
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

subplot(3,1,3);
plot(kernel(p_task,range));
title('Kernel Shape');