%%% history
%%%   - 2020/10/22 y.takagi - initially created with modifying Dmtry Kobak's dPCA program
%%%     see also: https://github.com/machenslab/dPCA
function [] = simulation_weightvisualize(seed)

%% Parameters
% general
addpath('../dPCA-master/matlab/')
rng(seed)
N_a_s = 5;
N_a_d = 5;
N_a = N_a_s+ N_a_d ;
N_b_s = 5;   
N_b_d = 4;
N_b = N_b_s + N_b_d;

T = 24;  % number of time points
S = 5;   % number of stimuli
D = 3;   % number of decisions
E = 20;  % maximal number of trial repetitions
noise_fac = 3.5;

time = (1:T) / 10;
delay_a2b = 4;
s_len = 10;
d_len = 8;
s_start_div = 4;
d_start_div = 2;
combinedParams = {{1}};

%% Generate Components
% Stimulus
component{1} = bsxfun(@times, ones(1,S,D,T,E), (1:S)-ceil(S/2)) .* ...
               bsxfun(@times, ones(1,S,D,T,E), shiftdim([time(1:end/s_start_div)*0 time(1:s_len) time(1:end-s_len-(end/s_start_div))*0], -2));

% Decision
component{2} = bsxfun(@times, ones(1,S,D,T,E), shiftdim((1:D)-ceil(D/2),-1)) .* ...
               bsxfun(@times, ones(1,S,D,T,E), shiftdim([time(1:end/d_start_div)*0 time(1:d_len) time(1:end-d_len-(end/d_start_div))*0], -2));            
           
%% Mixing components
% Region A
VV_a_s = randn(N_a_s, 1);
VV_a_s = bsxfun(@times, VV_a_s, 1./sqrt(sum(VV_a_s.^2)));
VV_b_d = randn(N_b_d, 1);
VV_b_d = bsxfun(@times, VV_b_d, 1./sqrt(sum(VV_b_d.^2)));
for c = 1:length(component)
    components(:,c) = component{c}(:,:);
end

% Region B
VV_a2b_s = randn(N_b_s,N_a_s);
VV_a2b_s = bsxfun(@times, VV_a2b_s, 1./sqrt(sum(VV_a2b_s.^2)));
VV_b2a_d = randn(N_a_d,N_b_d);
VV_b2a_d = bsxfun(@times, VV_b2a_d, 1./sqrt(sum(VV_b2a_d.^2)));

%% Firing rate
% region a
fr_a_s = VV_a_s * components(:,1)';
fr_a_s = fr_a_s + randn(size(fr_a_s))/noise_fac;

% region b
fr_b_d = VV_b_d * components(:,2)';
fr_b_d = fr_b_d + randn(size(fr_b_d))/noise_fac;

% region a -> region b
fr_a_s_rs =  reshape(fr_a_s, [N_a_s S D T E]);
input_s_a2b = fr_a_s_rs(:,:,:,T/s_start_div+1:T/s_start_div+s_len,:);
input_s_a2b = input_s_a2b(:,:);
input_s_a2b = VV_a2b_s*input_s_a2b;
input_s_a2b = reshape(input_s_a2b, [N_b_s, S, D, s_len, E]);

fr_b_s = zeros([N_b_s S D T E]);
fr_b_s(:,:,:,T/s_start_div+delay_a2b+1:T/s_start_div+delay_a2b+s_len,:) = input_s_a2b;
fr_b_s = fr_b_s + randn(size(fr_b_s))/noise_fac;
fr_b_s = fr_b_s(:,:);

% region b -> region a
fr_b_d_rs =  reshape(fr_b_d, [N_b_d S D T E]);
input_d_b2a = fr_b_d_rs(:,:,:,T/d_start_div+1:T/d_start_div+d_len,:);
input_d_b2a = input_d_b2a(:,:);
input_d_b2a = VV_b2a_d*input_d_b2a;
input_d_b2a = reshape(input_d_b2a, [N_a_d, S, D, d_len, E]);

fr_a_d = zeros([N_a_d S D T E]);
fr_a_d(:,:,:,T/d_start_div+delay_a2b+1:T/d_start_div+delay_a2b+d_len,:) = input_d_b2a;
fr_a_d = fr_a_d + randn(size(fr_a_d))/noise_fac;
fr_a_d = fr_a_d(:,:);

fr_a = [fr_a_s;fr_a_d]; % concatenate
fr_b = [fr_b_s;fr_b_d];

% normalizing
fr_a = fr_a - min(fr_a(:)) + 10;
fr_a = reshape(fr_a, [N_a S D T E]);

fr_b = fr_b - min(fr_b(:)) + 10;
fr_b = reshape(fr_b, [N_b S D T E]);

fr_a_s = permute(fr_a,[1,2,4,3,5]);   % Stimulus
fr_a_s = fr_a_s(:,:,:,:);
fr_a_d = permute(fr_a,[1,3,4,2,5]); % Decision
fr_a_d = fr_a_d(:,:,:,:);

fr_b_s = permute(fr_b,[1,2,4,3,5]);   % Stimulus
fr_b_s = fr_b_s(:,:,:,:);
fr_b_d = permute(fr_b,[1,3,4,2,5]); % Decision
fr_b_d = fr_b_d(:,:,:,:);

trial_nums_d(1,1:size(fr_a_d,2)) = size(fr_a_d,ndims(fr_a_d));
trial_nums_s(1,1:size(fr_a_s,2)) = size(fr_a_s,ndims(fr_a_s));

%% dSCA
err_d = zeros(T,T);
Ws_d = zeros(T,T, N_a_s + N_a_d);
Vs_d = zeros(T,T, N_b_s + N_b_d);

for ct1 = 1:T
    for ct2 = ct1+1:T
        t1 = ct1;
        t2 = ct2;
        [Ws_d(ct1,ct2,:), Vs_d(ct1,ct2,:), err_d(ct1,ct2)] = dsca_cv_noave_weightvisualize(fr_a_d(:,:,t1,:), fr_b_d(:,:,t2,:), trial_nums_d,trial_nums_d, ...
                        'combinedParams', combinedParams,'numComps',1, 'numRep', 25,'method','naive','simultaneous',true, 'lambda',0.1);
        [Vs_d(ct2,ct1,:), Ws_d(ct2,ct1,:), err_d(ct2,ct1)] = dsca_cv_noave_weightvisualize(fr_b_d(:,:,t1,:), fr_a_d(:,:,t2,:), trial_nums_d,trial_nums_d, ...
                        'combinedParams', combinedParams,'numComps',1, 'numRep', 25,'method','naive','simultaneous',true, 'lambda',0.1);   % or lambda = 1
    end
end

err_s = zeros(T,T);
Ws_s = zeros(T,T, N_a_s + N_a_d);
Vs_s = zeros(T,T, N_b_s + N_b_d);

for ct1 = 1:T
    for ct2 = ct1+1:T
        t1 = ct1;
        t2 = ct2;
        [Ws_s(ct1,ct2,:), Vs_s(ct1,ct2,:), err_s(ct1,ct2)] = dsca_cv_noave_weightvisualize(fr_a_s(:,:,t1,:), fr_b_s(:,:,t2,:), trial_nums_s,trial_nums_s, ...
                        'combinedParams', combinedParams,'numComps',1, 'numRep', 25,'method','naive','simultaneous',true, 'lambda',0.1);
        [Vs_s(ct2,ct1,:), Ws_s(ct2,ct1,:), err_s(ct2,ct1)] = dsca_cv_noave_weightvisualize(fr_b_s(:,:,t1,:), fr_a_s(:,:,t2,:), trial_nums_s,trial_nums_s, ...
                        'combinedParams', combinedParams,'numComps',1, 'numRep', 25,'method','naive','simultaneous',true, 'lambda',0.1); % or lambda = 1
    end
end

%% Error
figure('Renderer', 'painters', 'Position', [10 10 600 260])
subplot(1,2,1)
imagesc(-err_d,[-1,-0.5])
xlabel('Time of B')
ylabel('Time of A')
set(gca,'YDir','normal')
title('Decision')
colormap hot

subplot(1,2,2)
imagesc(-err_s,[-1,-0.5])
xlabel('Time of B')
ylabel('Time of A')
set(gca,'YDir','normal')
title('Stimulus')
colormap hot

idx_d = err_d>0.3 & err_d<0.95;
idx_s = err_s>0.3 & err_s<0.95;

%% Weights: Stimulus
figure('Renderer', 'painters', 'Position', [10 10 150 400])
for i = 1:size(Ws_s,3)
    subplot(5,2,i)
    im = Ws_s(:,:,i);
    im(~idx_s) = 0;
    imagesc(abs(im),[0,1])
    set(gca,'YDir','normal')
    colormap(hot)
end
fname = './figs/rebuttal_weight_X_Stimulus';
orient('landscape')
print('-bestfit',fname,'-dpdf') 

figure('Renderer', 'painters', 'Position', [10 10 150 400])
for i = 1:size(Vs_s,3)
    subplot(5,2,i)
    im = Vs_s(:,:,i);
    im(~idx_s) = 0;
    imagesc(abs(im),[0,1])
    set(gca,'YDir','normal')
    colormap(hot)
end
fname = './figs/rebuttal_weight_Y_Stimulus';
orient('landscape')
print('-bestfit',fname,'-dpdf') 


%% Weights: Decision
figure('Renderer', 'painters', 'Position', [10 10 150 400])
for i = 1:size(Ws_d,3)
    subplot(5,2,i)
    im = Ws_d(:,:,i);
    im(~idx_d) = 0;
    imagesc(abs(im),[0,1])
    set(gca,'YDir','normal')
    colormap(hot)
end
fname = './figs/rebuttal_weight_X_Decision';
orient('landscape')
print('-bestfit',fname,'-dpdf') 

figure('Renderer', 'painters', 'Position', [10 10 150 400])
for i = 1:size(Vs_d,3)
    subplot(5,2,i)
    im = Vs_d(:,:,i);
    im(~idx_d) = 0;
    imagesc(abs(im),[0,1])
    set(gca,'YDir','normal')
    colormap(hot)
end
fname = './figs/rebuttal_weight_Y_Decision';
orient('landscape')
print('-bestfit',fname,'-dpdf') 


end