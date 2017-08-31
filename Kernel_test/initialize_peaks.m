function [Xhat] = initialize_peaks( RY, theta, n, offset, Cy )
%% This function will determine which pixels to consider for analysis
% Inputs:
%	- RY: The lines under consideration
%	- DLt(RY,D,angles)

%-Add the path for original line_integral_adjoint function
addpath('../Chem_microscopy_code');

p_eps = 2e-1;
p_keep = 1;
SHOW_FIGURES = 0;

[line_dim, line_num] = size( RY );
theta = theta/180*pi; 
DLt = @(RY,theta, offset) ...
        fourier_line_integral_adjoint(RY, theta, n, offset, Cy);

X = zeros(n(1) ,n(2) , line_num);
%% For each line determine the adjoint
for i = 1:line_num
	Y = RY(:,i);

	dY = Y(2:end) - Y(1:end-1);					% First order
	ddY = dY(2:end) - dY(1:end-1);				% Second order
	dY_index = find(abs(dY)/norm(dY,2) > p_eps);
	ddY_index = find(ddY >= 0);
	Y_cand = Y;
	Y_cand(dY_index) = 0;
	Y_cand(ddY_index) = 0;
	%Y_cand(Y_cand > 0) = 1;

	X(:,:,i) = DLt( Y_cand, theta(i), offset(i) );
end

Xhat = sum(X,3);

% Keep a percentage of the pixels:
keep_num_pix = floor( p_keep * ( n(1) * n(2) ) );
Xhat_vals = reshape( Xhat, [1, n(1) * n(2)] );
Xhat_vals = sort( Xhat_vals, 'descend' );
threshold = Xhat_vals( keep_num_pix );
Xhat(Xhat < threshold) = 0;

%% Visualize the Results
if SHOW_FIGURES
	figure(10); clf;
	imagesc(Xhat);
	title('Selected Pixels')

	figure(11); clf;
	imagesc(DLt(RY, theta, offset))
	title('L*[RY]')

	%% Test to see what the linear transformation gives:
	RY_test = fourier_line_integral(Xhat,theta, 100,offset,Cy);
	RY_adj = fourier_line_integral(DLt(RY, theta, offset),theta, 100,offset,Cy);

	figure(12); clf;
	for i = 1:line_num
		subplot(2, ceil(line_num/2), i)
		hold on;
		plot(RY_test(:,i))
		plot(RY_adj(:,i))
		hold off;
	end
end