%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [W, V, whichMarg] = dsca_noave(Xtrial,Ytrial,numComps, varargin)

% default input parameters
options = struct('combinedParams', [],       ...   
                 'lambda',         0,        ...
                 'order',          'yes',    ...
                 'timeSplits',     [],       ...
                 'timeParameter',  [],       ...
                 'notToSplit',     [],       ...
                 'scale',          'no',     ...
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
Xt = Xtrial(:,:);
Xt = Xt(:,~all(isnan(Xt)));
XtCen = bsxfun(@minus, Xt, mean(Xt,2));

Yfull = nanmean(Ytrial,ndims(Ytrial));
Y = Yfull(:,:);
Y = bsxfun(@minus, Y, mean(Y,2));
YfullCen = reshape(Y, size(Yfull));


% total variance
totalVar = sum(Y(:).^2);

% marginalize
[Ymargs, margNums] = dpca_marginalize(YfullCen, 'combinedParams', options.combinedParams, ...
                    'timeSplits', options.timeSplits, ...
                    'timeParameter', options.timeParameter, ...
                    'notToSplit', options.notToSplit, ...
                    'ifFlat', 'yes');
                
                

for m = 1:length(Ymargs)
    YY = repmat(Ymargs{m},1,size(Ytrial,4));
    YY(:,all(isnan(Ytrial(:,:)))) = [];
    Ymargs{m} = YY;
end                
                
% initialize
decoder = [];
encoder = [];
whichMarg = [];

% noise covariance
if isempty(options.Cnoise)
    options.Cnoise = 0;
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
    
    C = Ymargs{i}*XtCen'*pinv(XtCen*XtCen' + options.Cnoise + (totalVar*thisLambda)^2*eye(size(XtCen,1)));
    
    M = C*XtCen;
    [U,~,~] = eigs(M*M', nc);
    P = U;
    D = U'*C;
    
    if strcmp(options.scale, 'yes')
        for uu = 1:size(D,1)
            A = Ymargs{i};
            B = P(:,uu)*D(uu,:)*XtCen;
            scalingFactor = (A(:)'*B(:))/(B(:)'*B(:));
            D(uu,:) = scalingFactor * D(uu,:);
        end
    end
    
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

    
end