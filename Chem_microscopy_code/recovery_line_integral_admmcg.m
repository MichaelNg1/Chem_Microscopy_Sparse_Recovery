function Result = recovery_line_integral_admmcg(RY,D,angles,n,initInput)
% [Description] Recovery of sparse chemical reactors in image domain
%  We observe the random projected convoluted signal as:
%     RY = L[D*X]   with   X sparse
%  Optimization problem: min |X|_1  s.t. Y = L[D*X]
%  Augmented Lagrangian: L(X,Z,M,Q) = |X|_1 + <M,X-Z> + <Q,Y-L[D*Z]> + 
%     mu1/2|X-Z|_F^2 + mu2/2|Y-L[D*Z]|_F^2
%  ADMM:
%     X <- argmin |X|_1 + mu1/2|X-Z+M/mu1|_F^2
%     Z <- argmin mu1/2|Z-X-M/mu1|_F^2 + mu2/2|L[D*Z]-Y-Q/mu2|_F^2
%     (M,Q) <- (M,Q) + (mu1(X-Z),mu2(Y-L[D*Z]))
%  Update Z with conjugate gradient:
%     Solve (mu1I + mu2LTd'LTd)Z = mu1X + M + LTd'(mu2Y+Q)  
%                     A        Z =           B
%     alpha    <- <R,R>/<P,AP>
%     Z(k)     <- Z + alpha*P
%     R(k)     <- R - alpha*AP
%     P(k)     <- R(k) + <R(k),R(k)>/<R,R>P
% 
% [Input]:
%   RY(nbins,nangles): Lines after integration
%   D:                 Kernel function              
%   angles(nangles,1): angles of line slopes
%   n(2,1):            [ny,nx]; size(Y)


%% ===== Parameter Definition ===== %%
LINE_INTEGRAL_METHOD = 'Fourier'; % 'Fourier', 'Direct'
niter_display_detail = 1;
niter_display_image = 1;
nAdmmIter = 1200;
residualError = 4e-4;
mu1 = 10.0;
mu2 = 10.0;
residualNormalizedErrorCG = 1e-5; 

[nbins, nangles] = size(RY);
m = n-size(D) + [1,1];

if strcmp(LINE_INTEGRAL_METHOD,'Direct')
    LD =  @(X,D,angles) line_integral(conv2(X,D),angles,nbins);
    LDt = @(RY,D,angles,n) conv2_adjoint(line_integral_adjoint(RY,angles,n),D);
elseif strcmp(LINE_INTEGRAL_METHOD,'Fourier')
    LD =  @(X,D,angles) fourier_line_integral(conv2(X,D),angles,nbins);
    LDt = @(RY,D,angles,n) conv2_adjoint(fourier_line_integral_adjoint(RY,angles,n),D);
end
La = @(X,Z,LDZ,M,Q) sum(sum(abs(X))) + sum(sum(Q.*(RY-LDZ))) + ...
                sum(sum(M.*(X-Z))) + mu1/2*norm(X-Z,'fro').^2 + ...
                mu2/2*norm(RY-LDZ,'fro').^2;
A = @(Z,D,angles,n,mu1,mu2) mu1*Z + mu2*LDt(LD(Z,D,angles),D,angles,n);
        
%-Adjoint of line integrals
LtRY = LDt(RY,D,angles,n);

%-Norm of RY
RYnorm = norm(RY,'fro');


%% ===== ADMM-CG ===== %%
if nargin == 5
    Xhat = initInput.Xhat;
    Zhat = initInput.Xhat;
    M = initInput.M;
    Q = initInput.Q;
else
    Xhat = zeros(m);
    Zhat = zeros(m);
    M = zeros(m);
    Q = zeros(nbins,nangles);
end
LDZ = LD(Zhat,D,angles);

iAdmmIter = 0;
kkk=0
while (norm(RY-LDZ,'fro')/RYnorm) > residualError && iAdmmIter < nAdmmIter
    T = Zhat-M/mu1;
    Xhat = sign(T).*max(abs(T)-1/mu1,0);
    
    %-Conjugate Gradient
    Zhat = zeros(m);
    B = mu1*Xhat + M + LDt( mu2*RY + Q ,D,angles,n);
    R = B;
    P = B;
    Rsold = sum(sum(R.*R));
    while norm(R,'fro')/sqrt(m(1)*m(2)) > residualNormalizedErrorCG
        A_P = A(P,D,angles,n,mu1,mu2);
        alpha = Rsold/sum(sum(P.*A_P)); %-Basis coefficient 
        Zhat = Zhat + alpha*P;          %-Unknown variable
        R = R - alpha*A_P;              %-Residual
        
        Rsnew = sum(sum(R.*R));         
        P = R + Rsnew/Rsold*P;          %-Conjugate basis
        Rsold = Rsnew;
    end
    LDZ = LD(Zhat,D,angles);
    M = M+mu1*(Xhat-Zhat);
    Q = Q+mu2*(RY-LDZ);
    
    if mod(iAdmmIter,niter_display_image) == 0
        fig = figure(2);
        fig.WindowStyle = 'docked';
        drawnow; colormap default;
        subplot(221); imagesc(LtRY); title('L^t[RY]'); axis equal; 
        subplot(222); imagesc(Xhat); title('X_{est}'); axis equal;
        subplot(223); imagesc(conv2(Xhat,D)); title('Y_{hat}'); axis equal;
        subplot(224); spy(abs(Xhat)>0.8); title('supp[X_{hat}]');
        
    end
    disp(['===== Number of Iteration : ', num2str(iAdmmIter), ' ====='])
    if mod(iAdmmIter,niter_display_detail) == 0
        disp(['L(Xhat,Zhat,M,Q) = ', num2str( La(Xhat,Zhat,LDZ,M,Q) ) ]);
        disp(['|RY-L[D*Zhat]|/|RY| = ', num2str(norm(RY-LDZ,'fro')/RYnorm) ]);
        disp(['|Xhat-Zhat| = ', num2str(norm(Xhat-Zhat,'fro'))])
        disp(['|R|/sqrt(m1*m2) = ' , num2str(norm(R,'fro')/sqrt(m(1)*m(2)))]);
    end;
    
    iAdmmIter = iAdmmIter + 1;
    if sum(iAdmmIter == [20,100,200,300,400,500,600,700,800,900,1000])
        kkk = kkk+1;
        Result(kkk).Xhat = Xhat;
    end
end
