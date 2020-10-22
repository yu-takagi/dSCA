%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [W, V, whichMarg] = dsca(Xfull,Yfull, numComps, varargin)

% default input parameters
options = struct('combinedParams', [],       ...   
                 'lambda',         0,        ...
                 'order',          'yes',    ...
                 'timeSplits',     [],       ...
                 'timeParameter',  [],       ...
                 'notToSplit',     [],       ...
                 'Cnoise',         []);

% read input parameters
optionNames = fieldnames(options);
if mod(length(varargin),2) == 1
	error('Please provide propertyName/propertyValue pairs')
end
for pair = reshape(varargin,2,[])    % pair is {propName; propValue}
	if any(strcmp(pair{1}, optionNames))
        options.(pair{1}) = pair{2};
    else
        error('%s is not a recognized parameter name', pair{1})
	end
end

% centering
X = Xfull(:,:);
X = bsxfun(@minus, X, mean(X,2));

Y = Yfull(:,:);
Y = bsxfun(@minus, Y, mean(Y,2));
YfullCen = reshape(Y, size(Yfull));


% total variance
totalVar = sum(Y(:).^2);
% totalVarY = sum(Y(:).^2);


% marginalize
[Ymargs, margNums] = dpca_marginalize(YfullCen, 'combinedParams', options.combinedParams, ...
                    'timeSplits', options.timeSplits, ...
                    'timeParameter', options.timeParameter, ...
                    'notToSplit', options.notToSplit, ...
                    'ifFlat', 'yes');
                
% initialize
decoder = [];
encoder = [];
whichMarg = [];

% noise covariance
if isempty(options.Cnoise)
    options.Cnoise = zeros(size(X,1));
end

% loop over marginalizations
for i=1:length(Ymargs)
    if length(numComps) == 1
        nc = numComps;
    else
        nc = numComps(margNums(i));
    end
    
    if length(options.lambda) == 1
        thisLambda = options.lambda;
    else
        thisLambda = options.lambda(margNums(i));
    end
    
    if nc == 0
        continue
    end
    
    C = Ymargs{i}*X'*pinv(X*X' + options.Cnoise + (totalVar*thisLambda)^2*eye(size(X,1)));
    
    M = C*X;
    [U,~,~] = eigs(M*M', nc);
    P = U;
    D = U'*C;
    
    decoder = [decoder; D];
    encoder = [encoder P];    
    whichMarg = [whichMarg i*ones(1, nc)];
end

% transposing
V = encoder;
W = decoder';

% flipping axes such that all encoders have more positive values
toFlip = find(sum(sign(V))<0);
W(:, toFlip) = -W(:, toFlip);
V(:, toFlip) = -V(:, toFlip);

% ordering components by explained variance (or not)
if length(numComps) == 1 || strcmp(options.order, 'yes')
    

    for i=1:size(W,2)
        Z = Y - V(:,i)*(W(:,i)'*X);
        explVar(i) = 1 - sum(Z(:).^2)/totalVar;
    end
    [~ , order] = sort(explVar, 'descend');
    
    if length(numComps) == 1
        L = numComps;
    else
        L = sum(numComps);
    end
    
    W = W(:, order(1:L));
    V = V(:, order(1:L));
    whichMarg = whichMarg(order(1:L));
end
    
end