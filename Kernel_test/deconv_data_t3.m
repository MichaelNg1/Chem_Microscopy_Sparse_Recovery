function [ ] = deconv_data_t3( RY )
%DECONV_DATA_T4 Summary of this function goes here
%   Detailed explanation goes here

%-Add the path for original lpsf function



%% Constants %%
[samples_num, lines_num] = size(RY);
range = 1:samples_num;
kernel = @lpsf;     % Kernel used
p_len = 3;
niter = 500;      
% Number of iterations
LAMBDA_SCALE = 0.1;

%-Integration factors
dp = 0.01 * ones(1,p_len);

%-Functions to generate observation and objective
CDf = @(kernel, range, p) convmtx(kernel(p, range), length(range));
gen_y = @(CDf, map) map * CDf;

dl = floor(samples_num/2) + 1;
dr = ceil(samples_num/2);
conv_resize = dl:(2*samples_num - dr);
objective = @(Yhat, Y) 0.5*norm(Yhat - Y,2)^2 + LAMBDA_SCALE * sum(abs(Yhat));

%- Initialize variables:
p_task = [.05, 2, -3];
x_task = zeros(1,samples_num) + .05;

%-Take a single line as truth
Y = RY(:,2)';
Y_long = [zeros(1, dl-1), Y, zeros(1, dr-1) ];

obj_adj = @(map, CDf) (map*CDf - Y_long) * CDf';

%-Gradient Step size initialization
% tp = 0.7/(norm(Y,'fro').^2);
tp_scale = 1 * ones(1,p_len);
CDf_init = CDf(kernel, range, p_task);

Z = randn(1, samples_num);
Z = Z/norm(Z,'fro');
for iiter = 1:20
    Z = (Z * CDf_init) * CDf_init';
    Z = Z/norm(Z,'fro'); 
end
Lip = 2*norm((Z * CDf_init) * CDf_init','fro');
tx = 1/Lip;

Yadj = Y_long * CDf_init';
lambda = LAMBDA_SCALE*max(max(abs(Yadj)));

%% (TASK 3) Estimate p and x %%
% For this task the objective function used will be:
% min 0.5 * || D(p) * x - y ||_2 ^2 + lambda ||x||_1
%       - d: kernel function
Yhat = zeros(1,samples_num);
error = zeros(1,niter);
for i = 1:niter
    CDf_task = CDf(kernel, range, p_task);
    Yhat_long = gen_y(CDf_task, x_task);
    Yhat = Yhat_long(1, conv_resize);

    e = objective(Yhat, Y);
    error(i) = e;
    
    %Estimate Jacobian
    Jp = cell(1,length(p_task));
    for k = 1:length(p_task)
        ek = zeros(size(p_task));
        ek(k) = 1;
        CDf_task_eps = CDf(kernel, range, p_task + dp.*ek);
        Yhat_eps_long = gen_y(CDf_task_eps, x_task);
        Yhat_eps = Yhat_eps_long(1, conv_resize);
        Jp{k} = (Yhat_eps - Yhat) / dp(k);
    end
    
    %Gradient step in p
    for k = 1:length(p_task)
        step = (tp_scale(k)/(norm(Jp{k},'fro')));
        p_task(k) = p_task(k) - step*sum(sum(Jp{k}.*(Yhat - Y)));
    end
    
    %Gradient step in x
    CDf_task = CDf(kernel, range, p_task); % Need to take into account updated p
    grad_f = obj_adj(x_task, CDf_task);
    x_task = x_task - tx*grad_f;
    x_task = sign(x_task).*max(abs(x_task)-lambda*tx,0);

    %Project onto correct subspace
    p_task(1) = max(1e-1, p_task(1));      %assuming lpsf
    p_task(2) = max(1e-1, p_task(2));      %assuming lpsf
    p_task(3) = min(-1e-1, p_task(3));     %assuming lpsf
    
    disp(['==== Number of iterations :', num2str(i), ' ====']);
    disp(['Objective: ', num2str(e)]);
    disp(['p_test: ', num2str(p_task)]);
    
    %Update Lambda
    CDf_init = CDf(kernel, range, p_task);
    Yadj = Y_long * CDf_init';
    lambda = LAMBDA_SCALE*max(max(abs(Yadj)));
    
    %Update tx
    if (mod(i,3) == 0)
        Z = randn(1, samples_num);
        Z = Z/norm(Z,'fro');
        for iiter = 1:20
            Z = (Z * CDf_init) * CDf_init';
            Z = Z/norm(Z,'fro'); 
        end
        Lip = 2*norm((Z * CDf_init) * CDf_init','fro');
        tx = 1/Lip;
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
plot(error);
xlabel('Number of Iterations');
ylabel('Error');
title('Objective Value');

subplot(4,1,3);
plot(kernel(p_task,range));
title('Kernel Shape');

subplot(4,1,4);
hold on;
stem(x_task);
hold off;
title('Learned Activation Map');
end

