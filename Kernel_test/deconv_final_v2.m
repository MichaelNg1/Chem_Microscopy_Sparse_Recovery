function [Xhat,params] = deconv_final_v1(RY, LD, D, params_init, dim,lambda) 
% [Description] 
%  Recovery of sparse chemical reactors in image domain
%  We observe the random projected convoluted signal as:
%         RY = D*L[X]   with   X sparse
%  Optimization problem: 
%      min_{X,p} lambda*|X|_1 + 1/2*|| RYtau - D(p)*L(p)[X] ||_F^2  -----(1)
%  Solve (1) with alternating proximal gradient descent
%
%  a. X   <- prox_{|.|_1/(t1*lanmbda)}[ X - t1*(DL)'(DLX - RY) ]
%  b. tau   <- tau - t2*Jtau'(RY-DLX)
%     theta <- theta - t3*Jtheta'(DLX-RY)
%     alpha <- alpha - t4*Jalpha'(RY-DLX)
%     p     <- p - tp.*Jp'(DLX-RY)
%
% [Input]:
%   RY(nbins,nangles): Lines after integration
%   params_init:  Initial parameters for optimization
%   dim:          Dimension parameter
%   lambda:       Penalty variable for lasso
%
% [Output]
%  Xhat(m):  Output sparse position
%  D      :  Kernel profile
%  params.amplitude:   amplitude for each lines in RY
%  params.rotation:    angle of slopes for each lines in RY
%  params.translation: transition for each lines in RY
%  params.p:           parameters 1D kernel shape
%  

%-Add the path for original lpsf function
addpath('../Chem_microscopy_code');

LINE_INTEGRAL_METHOD = 'Fourier';
LAMBDA_SCALE = 0.5;

if nargin == 5
    lambda = 0; % lambda placeholder
elseif nargin < 5 || nargin > 6
    error('Wrong number of input.');
end



%% ===== Input Parameters ===== %%
%-Dimension Variables
n = dim.n;
d = dim.d;
pixelLength = dim.PixelLength;
offset = dim.Offset;
Cy = dim.Cy;
direction = dim.Direction;

nbins = size(RY,1);
nlines = length(offset);

%-Integration Method & Integral interval
dtau = 1; %-[NOTE]-Should be integer
dtheta = 0.01;
dalpha = 0.01;
dp = [.01,.01,.01];

%-Gradient Parameters
CALIBRATE_TRANSLATION = 1;
CALIBRATE_ROTATION = 1;
CALIBRATE_SCALING = 1;
CALIBRATE_D = 1;

%-Update Function Parameters
UPDATE_LAMBDA = 1;
UPDATE_LIPSCHITZ = 1;

%-Enable Display Figure
PLOT_RECOVERED_FIG = 1;



%-Iteration Parameters
niter = 1000;
niterDisplayImage = 5;
niterDisplayDetail = 1;
niterUpdateLip = 50;
niterUpdateLambda = 50;

%-Variable Domain
tauRange = [-500,500]; %-tranlation (um)
thetaRange = [-15,15]; %-rotation (degree)

tauRange = tauRange/pixelLength;
thetaRange = thetaRange*pi/180;

thetaMid = params_init.Rotation/180*pi;


%% ===== Parameter Definition ===== %%
%-D index set 
didx = [1:nbins]';

%-Linear Operators and objective function
if strcmp(LINE_INTEGRAL_METHOD,'Direct')
    DL =  @(X,D,angles) line_integral(cropconv2(X,D),angles,nbins);
    DLt = @(RY,D,angles) cropconv2_adjoint(line_integral_adjoint(RY,angles,n),D);
elseif strcmp(LINE_INTEGRAL_METHOD,'Fourier')
    DL =  @(X,theta,D) ...
        convlines_pp(fourier_line_integral(X,theta,nbins,offset,Cy),D,direction);
    DLt = @(RY,theta,D) ...
        fourier_line_integral_adjoint(convlines_pp_adjoint(RY,D,direction),theta,n,offset,Cy);
end
f = @(RY,X,DLX,L) L*sum(sum(abs(X))) + 1/2*norm(RY - DLX,'fro')^2;


%-Variable Initilization
alpha = params_init.Amplitude;        %-Initial scaling
theta = params_init.Rotation/180*pi;  %-Initial angle of slopes
tau   = params_init.Translation;      %-Initial translation
p     = params_init.p;                %-Initial kernel parameter

% Use the parameter per pixel: 
p     = repmat(p, [ nbins, 1, nlines ]);
D_kernel = zeros( nbins, nbins, nlines );
for I = 1:nlines
    for i = 1:length(didx)
        D_kernel(:,i, I) = LD([p(i,:,I) didx(i)], didx);
    end
end

%-[TODO] Assumed kernel D via inverse Abel transform


%-Lipschtz of gradient of smooth function via power method
disp('Calculating Lipschitz Constant of Operator...');
Z = randn(nbins,nlines);
Z = Z/norm(Z,'fro');
for iiter = 1:20
    Z = DLt(DL(Z,theta,D_kernel),theta,D_kernel);
    Z = Z/norm(Z,'fro'); 
end
Lip = 2*norm(DLt(DL(Z,theta,D_kernel),theta,D_kernel),'fro');


%-Calculate lambda, L1 penalty variable
DLtRY = DLt(RY,theta,D_kernel);
if lambda == 0
    lambda = LAMBDA_SCALE*max(max(abs(DLtRY)));
end



%-Gradient Stepsize
tx = 1/Lip; %-Stepsize for X
ttau = 0.2/(norm(RY,'fro').^2); %-Stepsize for tau,theta
ttheta = ttau; 
talpha = ttau;
tp = ttau;






%% ===== PALM ===== %%
[Xhat, RY_INIT] = initialize_peaks( RY, theta, n, offset, Cy );
RY_MEANS = 1.75 * mean(RY_INIT);
DLXhat = zeros(size(RY));
RYtau = RY;

for iiter = 1:niter

    %---Recompute the kernel matrix for all lines ---%
    for I = 1:nlines
        for i = 1:length(didx)
            if(RY_INIT(i,I) > RY_MEANS(I))
                D_kernel(:,i, I) = LD([p(i,:,I) didx(i)], didx);
            end
        end
    end

    %---Assign new RYtau in new iteration---%
    if CALIBRATE_TRANSLATION
        for J = 1:nlines
            tt = tau(J);
            RYtau(:,J) = alpha(J)*imtranslate(RY(:,J),[0,tt]);
        end
    end
    
    if CALIBRATE_SCALING
        for J = 1:nlines
            a = alpha(J)*norm(RY(:,J),2)/norm(RYtau(:,J),2);
            RYtau(:,J) = a*RYtau(:,J);    
        end
    end
    

    %---Proximal Gradient for Xhat---%

    Xhat = Xhat - tx*DLt( DLXhat-RYtau, theta,D_kernel); 
    Xhat = sign(Xhat).*max(abs(Xhat)-lambda*tx,0);
    DLXhat = DL(Xhat,theta,D_kernel);

    RY_test = fourier_line_integral(Xhat,theta, n(1),offset,Cy);
    figure(14); clf;
    for i = 1:nlines
        subplot(2, ceil(nlines/2), i)
        hold on;
        plot(RY_test(:,i))
        hold off;
    end
    

    %---Jacobian---%
    if CALIBRATE_TRANSLATION 
        Jtau = [zeros(dtau,nlines) ; RYtau(1:end-dtau,:)] - RYtau;
    end
    if CALIBRATE_SCALING
        Jalpha = (RYtau*(1+dalpha)-RYtau)/dalpha;
    end
    if CALIBRATE_ROTATION
        Jtheta = (DL(Xhat,theta+dtheta,D_kernel)-DLXhat)/dtheta;
    end
    if CALIBRATE_D
        Jp = cell(1, nlines * nbins * size(p,2));

        parfor iteration = 1:(nlines * nbins * size(p,2))
            i = mod(iteration - 1, size(p,2)) + 1;                      % parameter
            j = mod(floor( (iteration - 1) / size(p,2) ), nbins) + 1;  % pixel
            k = floor( (iteration - 1) / (size(p,2) * nbins) ) + 1;     % line
            if (RY_INIT(j,k) > RY_MEANS(k))
                eK = zeros(1, size(p,2));
                eK(1,i) = 1;

                D_kernel_eps = D_kernel;
                D_kernel_eps(:, j, k) = LD([p(j, :, k)+dp.*eK didx(j)], didx);

                Jp{iteration} = (DL(Xhat,theta,D_kernel_eps) - DLXhat) / dp(i);
            end
        end
    end
    
    %---Gradient for tau, theta, p---%
    if CALIBRATE_TRANSLATION
        for J = 1:nlines
            tau(J) = tau(J) - ttau*Jtau(:,J)'*( RYtau(:,J) - DLXhat(:,J) );
        end
    end
    if CALIBRATE_SCALING
        for J = 1:nlines
            alpha(J) = alpha(J) - talpha*Jalpha(:,J)'*( RYtau(:,J) - DLXhat(:,J) );
        end
    end
    if CALIBRATE_ROTATION
        for J = 1:nlines 
            theta(J) = theta(J) - ttheta/2*Jtheta(:,J)'*( DLXhat(:,J) - RYtau(:,J) );
        end 
    end
    if CALIBRATE_D
        for iteration = 1:(nlines * nbins * size(p,2))
            if ( ~isempty( Jp{iteration} ) )
                j = mod(floor( (iteration - 1) / size(p,2) ), nlines) + 1;  % pixel
                k = floor( (iteration - 1) / (size(p,2) * nbins) ) + 1;     % line
                
                p(j, :, k) = p(j, :, k) - tp*sum(sum(Jp{iteration}.*(DLXhat - RYtau))); 
            end
        end
    end

    %---Projection to constraint set---%
    tau = max(tau,tauRange(1));
    tau = min(tau,tauRange(2));
    
    theta = max(theta,thetaMid+thetaRange(1));
    theta = min(theta,thetaMid+thetaRange(2));
    
    %-alpha in simplex * nangles
    alpha = max(alpha,0);
    alpha = min(alpha,nlines);
    alpha = alpha/sum(alpha)*nlines;
    
 
    %---Display Result---%
    disp(['===== Number of Iteration : ', num2str(iiter), ' ====='])
    if mod(iiter, niterDisplayDetail) == 0;
        disp(['tau(um) = ', num2str(tau*pixelLength)]);
        disp(['theta(deg) = ', num2str(theta*180/pi)]);
        disp(['alpha = ', num2str(alpha)]);
        % disp(['p = ', num2str(p)]);
        disp(['f = ', num2str( f(RYtau,Xhat,DLXhat,lambda) ) ]);
        disp(['|RYtau-D*L[Xhat]| = ', num2str(norm(RYtau-DLXhat,'fro')) ]);
    end
    
    %-Plot Result
    if PLOT_RECOVERED_FIG && mod(iiter,niterDisplayImage) == 0
        if iiter == niterDisplayImage
            fig = figure;
            fig.WindowStyle = 'docked';
            fig1 = get(fig,'Number');
            
            fig = figure;
            fig.WindowStyle = 'docked';
            fig2 = get(fig,'Number'); 
        end
        figure(fig1);
        drawnow;        
        LtRYtau = fourier_line_integral_adjoint(RYtau,theta,n,offset,Cy);
        subplot(221); subplot(221); imagesc(LtRYtau); title('L^t[RY]'); axis equal;
        subplot(222); imagesc(Xhat); title('X_{est}'); axis equal;
        subplot(223); imagesc(cropconv2(Xhat,D)); title('Y_{hat}'); axis equal;
        % subplot(224); plot(LD(p,didx)); title('D shape');
        
        figure(fig2);
        drawnow;
        nJ = min(nlines,10);
        for J = 1:min(nlines,10);
            subplot(nJ,2,2*J-1); plot(RYtau(:,J),'k');  title(['RY(\tau)[',num2str(J),']']);
            subplot(nJ,2,2*J);   plot(DLXhat(:,J));     title(['DLXhat[',num2str(J),']']);
        end
    end
    

    %---Update Lip/lambda
    if UPDATE_LIPSCHITZ && mod(iiter,niterUpdateLip) == 0
        disp('Calculateing Lipschitz Constant of Operator...');
        Z = randn(nbins,nlines);
        Z = Z/norm(Z,'fro');
        for J = 1:20
            Z = DLt(DL(Z,theta,p),theta,D_kernel);
            Z = Z/norm(Z,'fro'); 
        end
        Lip = 2*norm(DLt(DL(Z,theta,p),theta,D_kernel),'fro');
        tx = 1/Lip;
    end
    
    if UPDATE_LAMBDA && mod(iiter,niterUpdateLambda) == 0
        disp('Update lambda...')
        disp(['Previous lambda = ', num2str(lambda)]); 
        
        DLtRY = DLt(RY,theta,D_kernel);
        lambda = LAMBDA_SCALE*max(max(abs(DLtRY)));
        
        disp(['New lambda = ', num2str(lambda)]);
    end

    
    
end

params.Amplitude = alpha;
params.Rotation = theta*180/pi;
params.Translation = tau*pixelLength;
params.p = p;



