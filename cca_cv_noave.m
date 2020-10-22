%%% history
%%%   - 2020/10/22 y.takagi - initially created
function [meanerrors,meanccs] = cca_cv_noave(Xtrial,  Ytrial, numOfTrialsX, varargin)

% default input parameters
options = struct('numCompsCCA',       25,               ...   
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
    else
        Xtrainpc = Xtrain(:,:);
        Ytrainpc = Ytrain(:,:);
        Xtestpc = Xtest(:,:)';
        Ytestpc = Ytest(:,:)';
    end
    if options.zscore
        Xtrainpc = zscore(Xtrainpc);
        Ytrainpc = zscore(Ytrainpc);
        Xtestpc = zscore(Xtestpc);
        Ytestpc = zscore(Ytestpc);
    end
    
    %% CCA
    nd = size(Ytrainpc,2);
    if options.numCompsPCA > 0 && options.lambda > 0
        [A,B] = canoncorr(...
                        [Xtrainpc; options.lambda*eye(nd); zeros(nd)],...
                        [Ytrainpc; zeros(nd); options.lambda*eye(nd)]);    
    else
        [A,B] = canoncorr([Xtrainpc],[Ytrainpc]);  
    end
    
    %% Predict
    U = Xtestpc*A(:,1:options.numCompsCCA);
    V = Ytestpc*B(:,1:options.numCompsCCA);
    U_score  = U * pinv(B(:,1:options.numCompsCCA));    
        
    %% Error
    TestVar = sum(Ytestpc(:).^2);
    recError = sum(sum((Ytestpc - U_score).^2));
    errors(:,rep) = recError/TestVar;
    ccs(:,rep) = diag(corr(U,V));

end

meanerrors = mean(errors,2);
meanccs = mean(ccs(1,:),2);
