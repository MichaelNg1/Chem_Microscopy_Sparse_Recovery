function [Yhat, e] = construct_y(kernel, x, xi, p, Y)
%RECONSTRUCTION_ERROR is the l2 error of signal representation
%INPUT:
%OUTPUT:
Yhat = zeros(1,length(x));
for i = 1:length(xi)
    Yhat = Yhat + kernel([p xi(i)], x);
end

e = norm(Yhat - Y, 2)^2;

end

