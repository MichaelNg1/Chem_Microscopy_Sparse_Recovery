function y = lpsf_semi(p,x)
%-Convoluted semi-circle
% c * sqrt(a^2 - (x - b)^2) * (t+1)^n
%
% Input: p(1,3), [a,n]
%        p(1,4), [a,n,b]
%        p(1,5), [a,n,b,c]
%        x scalar, function domain


%-Input Parameter Allocation
xm = x(round(0.85*numel(x)/2)); %-Find 'mean' of x
if numel(p) == 5
    a=p(1); m=p(2); n=p(3); b=p(4); c=p(5);
elseif numel(p) == 4
    a=p(1); m=p(2); n=p(3); b=p(4);   c=1;
elseif numel(p) == 3
    a=p(1); m=p(2); n=p(3); b=xm;   c=1;
    error('Invalid Parameter Dimension')
end

%-Dimension Parameter
dd = x(2) - x(1); %-Pixel Width
s = 0:dd:max(x); %-Integrating variable


%-Convolution with integral
y = zeros(size(x));
for I = 1:numel(x)
    y(I) = c*sum( sqrt(max(0, m^2 - (m^2 / a^2) * (x(I) - b - s).^2)).*((s+1).^n)  )*dd;
end


%-Scaling y
if numel(p) ~= 5
    y = y/(sum(y)*dd); % Integral of y = 1
end
    
