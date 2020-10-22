function [meanerrors] = rrr_cv_noave(Xtrial, Ytrial, numOfTrialsX, varargin)

% default input parameters
options = struct('numCompsRRR',       25,               ...   
                 'numCompsPCA',       50,               ...   
                 'lambda',        0.5, ...
                 'numRep',         10,                  ...
                 'simultaneous',   false,               ...
                 'zscore',   false);

% read input parameters
optionNames = fieldnames(options);
for pair = reshape(varargin,2,[])    % pair is {propName; propValue}
	if any(strcmp(pair{1}, optionNames))
        options.(pair{1}) = pair{2};
    else
        error('%s is not a recognized parameter name', pair{1})
	end
end

for rep = 1:options.numRep
    %% Train/Test split
    [Xtest, Ytest, XtrainFull, YtrainFull] = dsca_getTestTrials_noave(Xtrial, Ytrial, min(numOfTrialsX), ...
        'simultaneous', options.simultaneous);    
    
    %% Exclude test trial
    Xtrain = XtrainFull(:,:)';
    Ytrain = YtrainFull(:,:)';
    ex_idx = all(isnan(Xtrain),2);
    Xtrain(ex_idx,:) = [];
    Ytrain(ex_idx,:) = [];
    
    %% PC
    if options.numCompsPCA > 0
        [coeff_x,Xtrainpc] = pca(Xtrain,'NumComponents',options.numCompsPCA);
        Xtestpc          = Xtest(:,:)'*pinv(coeff_x');

        [coeff_y,Ytrainpc] = pca(Ytrain,'NumComponents',options.numCompsPCA);
        Ytestpc          = Ytest(:,:)'*pinv(coeff_y');
        Xtrainpc = Xtrainpc';
        Xtestpc = Xtestpc';
        Ytrainpc = Ytrainpc';
        Ytestpc = Ytestpc';
    else
        Xtrainpc = Xtrain(:,:)';
        Ytrainpc = Ytrain(:,:)';
        Xtestpc = Xtest(:,:);
        Ytestpc = Ytest(:,:);
    end
    if options.zscore
        Xtrainpc = zscore(Xtrainpc);
        Ytrainpc = zscore(Ytrainpc);
        Xtestpc = zscore(Xtestpc);
        Ytestpc = zscore(Ytestpc);
    end
    
    %% RRR
    C = Ytrainpc*Xtrainpc'*pinv(Xtrainpc*Xtrainpc' + (options.lambda)^2*eye(size(Xtrainpc,1)));
    M = C*Xtrainpc;
    [U,~,~] = eigs(M*M', options.numCompsRRR);
    V = U;
    W = U'*C;
    W = W';
    
    %% Predict
    Ytestpc_pred = V*W'*Xtestpc;
        
    %% Error
    TestVar = sum(Ytestpc(:).^2);
    recError = sum(sum((Ytestpc - Ytestpc_pred).^2));
    errors(:,rep) = recError/TestVar;
end

meanerrors = mean(errors,2);
