function LX = convlines_adjoint(RY,D,dir)
% [Description]: 
%   Adjoint operator of D*L[X]
DIRECTION_FORWARD = 1;
DIRECTION_BACKWARD = -1;
[~,nangles] = size(RY);

LX = zeros(size(RY));
for I = 1:nangles

	if dir(I) == DIRECTION_FORWARD
        LXI = D(:,:,I)' * RY(:,I);
    elseif dir(I) == DIRECTION_BACKWARD
        LXI = flipud(D(:,:,I))' * RY(:,I);
    end
    
    LX(:,I) = LXI;
end




