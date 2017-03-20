%% TODO: Make a well annotated function from this script

clear all;
%-Add the path for original lpsf function
addpath('C:\Users\Michael\Documents\Columbia\Wright\Line Detection Code\Chem_microscopy_code');

%% Constants %%
samples_num = 128;  % Number of samples
kernel = @lpsf;     % Kernel used
niter = 100;         % Number of iterations
k_num = 7;          % Number of "spikes"

%% Generate Artificial Test Data %%
%-Generate artificial activation map
xi = ceil(samples_num * rand(1,k_num));
x_mag = ones(1,numel(xi));
range = 1:samples_num;          % assume unit distance
activation_map = zeros(1, samples_num);
activation_map(xi) = 1;

%-Genearte random parameters
p = [1, 2, -2];

%-Generate observation
Y = zeros(1, samples_num);
for i = 1:numel(xi)
    Y = Y + kernel([p xi(i)],range);
end

%-Integration factors
dp = 0.01 * ones(1,3);

%-Step size initialization
tp = 0.5/(norm(Y,'fro').^2); % Think about this

objective = @(kernel, x, xi, p, Y) construct_y(kernel, x, xi, p, Y);

%% (TASK 1) Estimate p given xi %%
% For this task the objective function used will be:
% min || sum_(xi) d(x - xi)(p) - y ||_2
%       - d: kernel function

%- Initialize variables:
p_task = zeros(1,3) + 1;
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    [Yhat,e] = construct_y(kernel, range, xi, p_task, Y);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(1,length(p_task));
    for k = 1:length(p_task)
        ek = zeros(size(p_task));
        ek(k) = 1;
        [Yhat_eps, e_eps] = construct_y(kernel, range, xi, p_task + dp.*ek, Y);
        Jp{k} = (Yhat_eps - Yhat) / dp(k);
    end
    
    %Gradient step
    for k = 1:length(p_task)
        p_task(k) = p_task(k) - tp*sum(sum(Jp{k}.*(Yhat - Y)));
    end
    
    %Project onto correct subspace
    p_task(1) = max(1e-14, p_task(1));      %assuming lpsf
    p_task(2) = max(1e-14, p_task(2));      %assuming lpsf
    
    %-TODO: Update stepsize
    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    disp(['p: ', num2str(p_task)]);
end

%% Visualization %%
figure(1); clf;

subplot(3,1,1);
hold on;
plot(range, Y);
plot(range, Yhat);
legend('Truth', 'Learned');
scatter(xi,zeros(1,k_num),'x');
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

