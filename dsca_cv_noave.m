%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [meanError] = dsca_cv_noave(Xtrial, Ytrial, numOfTrialsX, numOfTrialsY, varargin)

% default input parameters
options = struct('numComps',       25,                  ...   
                 'lambda',         1e-10, ...
                 'numRep',         10,                  ...
                 'combinedParams', [],                  ...
                 'method',         'naive',          ...
                 'simultaneous',   false,               ...
                 'numPC',          [],                  ...
                 'order',          'no');

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

if min(numOfTrialsX(:)) <= 0 || min(numOfTrialsY(:)) <= 0 
    error('dSCA:tooFewTrials0','Some neurons seem to have no trials in some condition(s).')
elseif min(numOfTrialsX(:)) == 1 || min(numOfTrialsY(:)) == 1
    error('dSCA:tooFewTrials1','Cannot perform cross-validation.')
end

if ~isempty(options.numPC)
    [~,XX] = pca(Xtrial(:,:)', 'NumComponents', options.numPC);
    XX = XX';
    Xtrial = reshape(XX,[options.numPC,size(Xtrial,2),size(Xtrial,3),size(Xtrial,4)]);
end

for rep = 1:options.numRep
    [Xtest, Ytest, XtrainFull, YtrainFull] = dsca_getTestTrials_noave(Xtrial, Ytrial, min(numOfTrialsX), ...
        'simultaneous', options.simultaneous);
    
    XtestCen = bsxfun(@minus, Xtest, mean(Xtest(:,:),2));
    YtestCen = bsxfun(@minus, Ytest, mean(Ytest(:,:),2));
    YtestMargs = dpca_marginalize(YtestCen, 'combinedParams', options.combinedParams, ...
                    'ifFlat', 'yes');
    for i=1:length(YtestMargs)
        margTestVar(i) = sum(YtestMargs{i}(:).^2);
    end
    
    margVar_toNormalize = margTestVar;
        
    [W,V,whichMarg] = dsca_noave(XtrainFull,  YtrainFull,...
        options.numComps, ...
        'combinedParams', options.combinedParams, ...
        'lambda',options.lambda, 'order',options.order);

    for i=1:length(YtestMargs)
        recError = 0;

        if strcmp(options.method, 'naive')
            recError = sum(sum((YtestMargs{i} - V(:,whichMarg==i)*W(:,whichMarg==i)'*XtestCen(:,:)).^2));
        end

        errorsMarg(i, rep) = recError/margVar_toNormalize(i);
    end
end


meanError = mean(errorsMarg,2);
