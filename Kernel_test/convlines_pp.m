function RY = convlines(LX,D,dir)
%% 
% [Description]:
%    Convolution on lines for asymmetric D
DIRECTION_FORWARD = 1;
DIRECTION_BACKWARD = -1;
[~,nangles] = size(LX);

RY = zeros(size(LX));
for I = 1:nangles

	% if dir(I) == DIRECTION_FORWARD
        RYI = D(:,:,I) * LX(:,I);
    % elseif dir(I) == DIRECTION_BACKWARD
    %     RYI = flipud(D(:,:,I)) * LX(:,I);
    % end

    RY(:,I) = RYI;
end





