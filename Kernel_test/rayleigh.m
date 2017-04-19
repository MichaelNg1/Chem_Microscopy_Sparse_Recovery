function [ output_args ] = rayleigh(p, x)
%RAYLEIGH Summary of this function goes here
%   Detailed explanation goes here
%Standard Form: x/(sigma^2) exp{- x^2/(2 sigma^2) }
%Input: p(1) = [sigma]
%       x scalar, function domain

if numel(p) == 1
    sigma = p(1);
else
    error('Invalid Parameter Dimension');
end



end

