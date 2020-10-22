%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [Xtest, Ytest, Xtrain, Ytrain] = dsca_getTestTrials_noave(firingRatesPerTrialX, firingRatesPerTrialY, numOfTrials, varargin)

options = struct('simultaneous', false);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dimX = size(firingRatesPerTrialX);
dimY = size(firingRatesPerTrialY);

if ~options.simultaneous
    neuronsConditions = numOfTrials(:);
    testTrials = ceil(rand([length(neuronsConditions) 1]) .* neuronsConditions);
else
    neuronsConditions = numOfTrials(:);
    neuronsConditions = neuronsConditions(1:size(numOfTrials,1):end); 
    testTrials = ceil(rand([length(neuronsConditions) 1]) .* neuronsConditions);
    testTrials = bsxfun(@times, ones(size(numOfTrials,1),1), testTrials');
    testTrials = testTrials(:);
end

indX = reshape(testTrials, size(numOfTrials));
indX = bsxfun(@times, ones(dimX(1:end-1)), indX);
indX = indX(:);

indY = reshape(testTrials, size(numOfTrials));
indY = bsxfun(@times, ones(dimY(1:end-1)), indY);
indY = indY(:);

indtestX = sub2ind([prod(dimX(1:end-1)) dimX(end)], (1:prod(dimX(1:end-1)))', indX);

indtestY = sub2ind([prod(dimY(1:end-1)) dimY(end)], (1:prod(dimY(1:end-1)))', indY);

Xtest = firingRatesPerTrialX(indtestX);
Xtest = reshape(Xtest, dimX(1:end-1));

Ytest = firingRatesPerTrialY(indtestY);
Ytest = reshape(Ytest, dimY(1:end-1));


if nargout > 2
    Xtrain = firingRatesPerTrialX;
    Xtrain(indtestX) = nan;

    Ytrain = firingRatesPerTrialY;
    Ytrain(indtestY) = nan;
end
