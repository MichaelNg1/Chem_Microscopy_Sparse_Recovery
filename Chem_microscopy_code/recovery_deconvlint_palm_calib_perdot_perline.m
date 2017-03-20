function [Xhat,LD,params] = recovery_deconvlint_palm_calib_perdot_perline...
         (RY,D,LD,paramsInit,dim,lambda) 
% [Description] 
%  Recovery of sparse chemical reactors in image domain
%  We observe the random projected convoluted signal as:
%         RYi = P[ sum_j Dij(p)*Li[Xj] ]    with   X sparse
%  where 1. RYi is observed lines with slope angle theta_i
%        2. Xj is a single activation map, or a single dot.
%        3. Dij is PSF each line each dot, parameterized with p
%        4. P is cropping operator on each line.
%
%  We propose to solve the following optimization problem: 
%      min_{X,p} lambda*|X|_1 + 1/2*sum_i|| RYi(p) - P[ sum_j Dij(p)*Li(p)[Xj] ] ||_F^2  -----(1)
%  with alternating proximal gradient descent
%  1. X   <- prox_{|.|_1/(t1*lanmbda)}[ X - t1* Ai(p)'(Ai(p)[X] - RYi) ]
%  2. tau <- tau - t2*Jtau'(DLX-RY)
%     theta <- theta - t3*Jtheta'(DLX-RY)
%     rad <- rad - t4*Jrad'(DLX-RYtau)
%     sigma <- sigma - t5*Jsigma'*(DLX - RYtau)
%     alpha <- alpha - t6*Jalpha'*(RYtau-DLX)
%
% [Input]:
%   RY(nbins,nangles): Lines after integration
%   params_init:  Initial angles of line slopes
%   dim:          Dimension parameter
%   lambda:       Penalty variable for lasso
%
% [Output]
%  Xhat(m):  Output sparse position
%  D      :  Kernel profile
%  params.amplitude:   amplitude for each lines in RY
%  params.rotation:    angle of slopes for each lines in RY
%  params.translation: transition for each lines in RY
%  params.diameter:    diameter for the kernel profile D
%  params.sigma:       smoothing parameter for kernel profile D

if nargin < 4
    error('Insufficient input.');
end

LINE_INTEGRAL_METHOD = 'Fourier';


%% ===== Input Parameters ===== %%
%-Dimension Variables
n = dim.n;
pixelLength = dim.pixelLength;
offset = dim.offset;
Cy = dim.Cy;
direction = dim.Direction;

nbins = size(RY,1);
nangles = length(offset);


%-Integration Method & Integral interval
dtau = 3; %-[NOTE]-Should be integer
dtheta = 0.01;
dalpha = .1;


%-Gradient Parameters
CALIBRATE_TRANSLATION = 1;
CALIBRATE_ROTATION = 1;
CALIBRATE_SCALING = 1;


%-Iteration Parameters
niter = 2000;
niterDisplayImage = 10;
niterDisplayDetail = 1;


%-Variable Domain
tauRange = [-500,500]; %-tranlation (um)
thetaRange = [-15,15]; %-rotation (degree)

tauRange = tauRange/pixelLength;
thetaRange = thetaRange*pi/180;

thetaInit = paramsInit.Rotation/180*pi;


%% ===== Parameter Definition ===== %%

%-Linear Operators and objective function
if strcmp(LINE_INTEGRAL_METHOD,'Direct')
    DL =  @(X,D,angles) line_integral(cropconv2(X,D),angles,nbins);
    DLt = @(RY,D,angles,n) cropconv2_adjoint(line_integral_adjoint(RY,angles,n),D);
elseif strcmp(LINE_INTEGRAL_METHOD,'Fourier')
    DL =  @(X,D,angles) convlines(fourier_line_integral(X,angles,nbins,offset,Cy),D,direction);
    DLt = @(RY,D,angles,n) fourier_line_integral_adjoint(convlines_adjoint(RY,D,direction),angles,n,offset,Cy);
end
f = @(RY,X,DLX,L) L*sum(sum(abs(X))) + 1/2*norm(RY - DLX,'fro')^2;


%-Variable Initilization
alpha = paramsInit.Amplitude;        %-Initial scaling
theta = paramsInit.Rotation/180*pi;  %-Initial angle of slopes
tau   = paramsInit.Translation;      %-Initial translation


%-Lipschtz of gradient of smooth function
Xe = rand(n);
Xe = Xe/norm(Xe,'fro');
Lip = 20*norm(DLt(DL(Xe,LD,theta),LD,theta,n));
t1 = 1/Lip; %-Stepsize for X
t2 = 0.2/(norm(RY,'fro').^2); %-Stepsize for tau,theta


%-L1-penalty variable
if nargin ~= 6
    lambda = .5*max(max(abs(DLt(RY,LD,thetaInit,n))));
end


%% ===== PALM ===== %%
Xhat = zeros(n);
DLXhat = zeros(size(RY));
RYtau = RY;

for I = 1:niter
    
    %---Assign RYtau in new iteration---%
    if CALIBRATE_TRANSLATION
        for J = 1:nangles
            tt = tau(J);
            RYtau(:,J) = alpha(J)*imtranslate(RY(:,J),[0,tt]);
        end
    end
    
    if CALIBRATE_SCALING
        for J = 1:nangles
            a = alpha(J)*norm(RY(:,J),2)/norm(RYtau(:,J),2);
            RYtau(:,J) = a*RYtau(:,J);    
        end
    end
    
    %---Proximal Gradient for Xhat---%
    Xhat = Xhat - t1*DLt( DLXhat - RYtau,LD,theta,n ); 
    Xhat = sign(Xhat).*max(abs(Xhat)-lambda*t1,0);
    DLXhat = DL(Xhat,LD,theta);
    
    %---Jacobian---%
    if CALIBRATE_TRANSLATION 
        Jtau = [zeros(dtau,nangles) ; RYtau(1:end-dtau,:)] - RYtau;
    end
    if CALIBRATE_SCALING
        Jalpha = (RYtau*(1+dalpha)-RYtau)/dalpha;
    end
    if CALIBRATE_ROTATION
        Jtheta = (DL(Xhat,LD,theta+dtheta)-DLXhat)/dtheta;
    end

    %---Gradient for tau, theta, diam, dsigma---%
    if CALIBRATE_TRANSLATION
        for J = 1:nangles
            tau(J) = tau(J) - t2*Jtau(:,J)'*( RYtau(:,J) - DLXhat(:,J) );
        end
    end
    if CALIBRATE_SCALING
        for J = 1:nangles
            alpha(J) = alpha(J) - t2*Jalpha(:,J)'*(RYtau(:,J) - DLXhat(:,J));
        end
    end
    if CALIBRATE_ROTATION
        for J = 1:nangles 
            theta(J) = theta(J) - t2/2*Jtheta(:,J)'*( DLXhat(:,J) - RYtau(:,J) );
        end 
    end 


    %---Projection to constraint set---%
    tau = max(tau,tauRange(1));
    tau = min(tau,tauRange(2));
    
    theta = max(theta,thetaInit+thetaRange(1));
    theta = min(theta,thetaInit+thetaRange(2));
    
    
    %-alpha in simplex * nangles
    alpha = max(alpha,0);
    alpha = min(alpha,nangles);
    alpha = alpha/sum(alpha)*nangles;
    
   
    if mod(I,niterDisplayImage) == 0
        if I == niterDisplayImage
            fig = figure;
            fig.WindowStyle = 'docked';
            figNumber = get(fig,'Number');
        end
        figure(figNumber);
        drawnow;        
        LtRYtau = fourier_line_integral_adjoint(RYtau,theta,n,offset,Cy);
        subplot(221); subplot(221); imagesc(LtRYtau); title('L^t[RY]'); axis equal;
        subplot(222); imagesc(Xhat); title('X_{est}'); axis equal;
        subplot(223); imagesc(cropconv2(Xhat,D)); title('Y_{hat}'); axis equal;
        subplot(224); plot(LD); title('D shape');
    end
    
    disp(['===== Number of Iteration : ', num2str(I), ' ====='])
    if mod(I, niterDisplayDetail) == 0;
        disp(['tau(um) = ', num2str(tau*pixelLength)]);
        disp(['theta(deg) = ', num2str(theta*180/pi)]);
        disp(['alpha = ', num2str(alpha)]);
        disp(['f = ', num2str( f(RYtau,Xhat,DLXhat,lambda) ) ]);
        disp(['|RYtau-D*L[Xhat]| = ', num2str(norm(RYtau-DLXhat,'fro')) ]);      
    end
    
end

params.Amplitude = alpha;
params.Rotation = theta*180/pi;
params.Translation = tau*pixelLength;



