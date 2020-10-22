%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [meanError] = dsca_cv_bothmerge(Xfull, ...
    Xtrial, Yfull, Ytrial, numOfTrialsX, numOfTrialsY, varargin)


% default input parameters
options = struct('numComps',       25,                  ...   
                 'lambda',         1e-10, ...
                 'numRep',         10,                  ...
                 'display',        'yes',               ...
                 'combinedParams', [],                  ...
                 'method',         'naive',          ...
                 'simultaneous',   false);

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
    error('dSCA:tooFewTrials0','Some neurons seem to have no trials in some condition(s)')
elseif min(numOfTrialsX(:)) == 1 || min(numOfTrialsY(:)) == 1
    error('dSCA:tooFewTrials1','Cannot perform cross-validation')
end

Xsum = bsxfun(@times, Xfull, numOfTrialsX);
Ysum = bsxfun(@times, Yfull, numOfTrialsY);

for rep = 1:options.numRep
    
    if options.simultaneous
        [Xtest, Ytest, ~, ~] = dsca_getTestTrials_noave(Xtrial, Ytrial, min(numOfTrialsX), ...
            'simultaneous', options.simultaneous);
    else
        [Xtest, ~] = dpca_getTestTrials(Xtrial, numOfTrialsX, ...
            'simultaneous', options.simultaneous);
        [Ytest, ~] = dpca_getTestTrials(Ytrial, numOfTrialsY, ...
            'simultaneous', options.simultaneous);
    end
    Xtrain = bsxfun(@times, Xsum - Xtest, 1./(numOfTrialsX-1));
    Ytrain = bsxfun(@times, Ysum - Ytest, 1./(numOfTrialsY-1));
        
    XtestCen = bsxfun(@minus, Xtest, mean(Xtest(:,:),2));
    XtestMargs = dpca_marginalize(XtestCen, 'combinedParams', options.combinedParams, ...
                    'ifFlat', 'yes');
    YtestCen = bsxfun(@minus, Ytest, mean(Ytest(:,:),2));
    YtestMargs = dpca_marginalize(YtestCen, 'combinedParams', options.combinedParams, ...
                    'ifFlat', 'yes');
    for i=1:length(YtestMargs)
        margTestVar(i) = sum(YtestMargs{i}(:).^2);
    end
    
    if strcmp(options.method, 'naive')
        margVar_toNormalize = margTestVar;
    end

        
    [W,V,whichMarg] = dsca_bothmerge(Xtrain, Ytrain, options.numComps, ...
        'combinedParams', options.combinedParams, ...
        'lambda',options.lambda);

    for i=1:length(YtestMargs)
        recError = 0;

        if strcmp(options.method, 'naive')
            recError = sum(sum((YtestMargs{i} - V(:,whichMarg==i)*W(:,whichMarg==i)'*XtestMargs{i}).^2));
        end
        errorsMarg(i, rep) = recError/margVar_toNormalize(i);
    end
end


meanError = mean(errorsMarg,2);