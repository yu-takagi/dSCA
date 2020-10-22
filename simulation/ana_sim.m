function [] = ana_sim(use_var)
    %% Parameters
    nseed_for_perm = 1000;
    seed_for_real = 1;
    datdir = './results/simulation';
    var_lists = {'cc_noave','err_rrr_noave','err_s','err_d','err_bm_s','err_bm_d','err_cca_noave'};
    thres_cc = 0.4;
    thres_er = 0.95;
    
    %% Load
    disp('Loading actual data...')
    if exist([datdir,filesep,num2str(seed_for_real,'%04d'),'.mat'])
        load([datdir,filesep,num2str(seed_for_real,'%04d'),'.mat']) % load 'out'
    else
        out = simulation(seed_for_real);
    end
    
    disp('Loading permutation data...')
    for seed = 1:nseed_for_perm
        out_perm(seed) = load([datdir,'_perm',filesep,num2str(seed,'%04d')]);
    end
    
    %% Detect cluster (actual)
    dum = out.(var_lists{use_var});
    dum(logical(eye(size(dum)))) = NaN;
    if strcmp(var_lists{use_var}(1:2),'cc')
        [labeledA_true, numRegions_true] = bwlabel(dum>thres_cc, 4);
    elseif strcmp(var_lists{use_var}(1:3),'err')
        [labeledA_true, numRegions_true] = bwlabel(dum<thres_er, 4);
    end
    
    %% Detect max cluster size (perm)
    max_cc = 0;
    for seed = 1:nseed_for_perm
        dum = out_perm(seed).out.(var_lists{use_var});
        dum(logical(eye(size(dum)))) = NaN;
        if strcmp(var_lists{use_var}(1:2),'cc')
            [labeledA_perm, numRegions_perm] = bwlabel(dum>thres_cc, 4);
        else
            [labeledA_perm, numRegions_perm] = bwlabel(dum<thres_er, 4);
        end
        clst_size = [];
        for i = 1:numRegions_perm
            clst_size(i)=length(find(labeledA_perm==i));
        end
        max_cc = max(max_cc,max(clst_size));
    end
    if isempty(max_cc)
        max_cc = 0;
    end
    
    %% Determine significance
    sig_map = false(size(labeledA_true));
    for i = 1:numRegions_true
        clst_size=length(find(labeledA_true==i));
        if clst_size > max_cc
            sig_map(labeledA_true == i) = true;
        end
    end
    
    %% Make figure
    figure('Renderer', 'painters', 'Position', [10 10 300 260])
    im = out.(var_lists{use_var});
    im(~sig_map) = NaN;
    im(logical(eye(size(im)))) = NaN;
    if strcmp(var_lists{use_var}(1:2),'cc')
        imagesc(im,[0,1])
    elseif strcmp(var_lists{use_var}(1:3),'err')
        imagesc(1-im,[0,0.5])
    end
    title(var_lists{use_var},'Interpreter','none')
    xlabel('Time of B')
    ylabel('Time of A')
    set(gca,'YDir','normal')
    colormap hot
    colorbar()
    saveas(gcf,['./figs/simulation_',var_lists{use_var},'.pdf'])

end
