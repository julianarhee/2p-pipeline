function [options, nmf_outpaths] = getRois3Dnmf(D, meta, show_plots, getref)
% clear all;
% clc;
% 
% addpath(genpath('~/Repositories/ca_source_extraction'));
% addpath(genpath('~/Repositories/NoRMCorre'));

%% Specify tiff sources and covnert to matfile objs:

if isfield(D.maskInfo, 'nmfPaths')
    nmf_outpaths = D.maskInfo.nmfPaths;
else
    nmf_outpaths = {};
end

if D.average
    mempath = D.averagePath;
else
    %mempath = fullfile(D.nmfPath, 'memfiles');
    mempath = D.mempath;
end

tmpfiles = dir(fullfile(mempath, '*.mat'));
tmpfiles = {tmpfiles(:).name}';
if ~D.processedtiffs
    subidxs = cell2mat(cellfun(@(x) ~isempty(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0));
    files = tmpfiles(subidxs);
else
    files = tmpfiles;
end

if D.maskInfo.patches
    patch_size = D.maskInfo.params.patch_size;
    overlap = D.maskInfo.params.overlap;
end
K = D.maskInfo.params.K;
tau = D.maskInfo.params.tau;
p = D.maskInfo.params.p;
merge_thr = D.maskInfo.params.merge_thr;
 
for tiffidx = 1:length(files)
  
   
    if ~getref && tiffidx~=D.maskInfo.ref.tiffidx

        usePreviousA = true;

    elseif getref && tiffidx~=D.maskInfo.ref.tiffidx

        fprintf('Skipping... tiff idx: %i.\n', tiffidx)
        continue;

    elseif getref && tiffidx==D.maskInfo.ref.tiffidx
        fprintf('Getting REF components! Specified ref tiff is %i.\n', D.maskInfo.ref.tiffidx)
        usePreviousA = false;

    end
 
    first_tic = tic();

    tpath = fullfile(mempath, files{tiffidx});
    [filepath, filename, ext] = fileparts(tpath);

    matpath = fullfile(mempath, sprintf('%s.mat', filename));
    data = matfile(matpath,'Writable',true);

    fprintf('Processing components for FILE %s (%i of %i tiffs).\n', filename, tiffidx, length(files));


    nSlices = meta.file(tiffidx).si.nFramesPerVolume;
    nRealFrames = meta.file(tiffidx).si.nSlices;
    nVolumes = meta.file(tiffidx).si.nVolumes;
    if D.tefo
        nChannels=2;
    else
        nChannels=1;
    end

    if ndims(data.Y) == 4
        [d1,d2,d3,T] = size(data.Y)                            % dimensions of dataset
    else
        [d1,d2,T] = size(data.Y);
        d3 = 1;
    end
    d = d1*d2*d3;                                          % total number of pixels


%% Test NoRMCorre moition correction:

    MC = false;

% % Rigid:
% 
% % set parameters:
% options_rigid = NoRMCorreSetParms('d1',size(data.Y,1),'d2',size(data.Y,2),'d3',size(data.Y,3),...
%                 'bin_width',50,'max_shift',15,'us_fac',50);
% % perform motion correction
% tic; [M1,shifts1,template1] = normcorre(data,options_rigid); toc
% 
% % Save MC outputs:
% savefast([filepath(1:end-4),'_MC.mat'],'M1','shifts1','template1');           % save shifts of each file at the respective subfolder
% mcdata = matfile([filepath(1:end-4),'_MC.mat'], 'Writable', true);
% 
% 
% % ** this gets killed.... **
% % now try non-rigid motion correction (also in parallel)
% options_nonrigid = NoRMCorreSetParms('d1',size(data.Y,1),'d2',size(data.Y,2),'d3',size(data.Y,3),...
%                 'grid_size',[32,32],'mot_uf',4,'bin_width',50,'max_shift',15,...
%                 'max_dev',3,'us_fac',50);
% tic; [M2,shifts2,template2] = normcorre_batch(data,options_nonrigid); toc
% 
% %% compute metrics
% 
% nnY = quantile(data.Y(:),0.005);
% nnY = min(nnY(:));
% mmY = quantile(data.Y(:,:,:,:),0.995);
% mmY = min(mmY(:));
% 
% [cY,mY,vY] = motion_metrics(data.Y,5); %10);
% [cM1,mM1,vM1] = motion_metrics(M1,5); %10);
% 
% [cM2,mM2,vM2] = motion_metrics(M2,10);
% T = length(cY);
% 
% %% plot metrics
% figure;
%     ax1 = subplot(2,3,1); imagesc(mY,[nnY,mmY]);  axis equal; axis tight; axis off; title('mean raw data','fontsize',14,'fontweight','bold')
%     ax2 = subplot(2,3,2); imagesc(mM1,[nnY,mmY]);  axis equal; axis tight; axis off; title('mean rigid corrected','fontsize',14,'fontweight','bold')
%     ax3 = subplot(2,3,3); imagesc(mM2,[nnY,mmY]); axis equal; axis tight; axis off; title('mean non-rigid corrected','fontsize',14,'fontweight','bold')
%     subplot(2,3,4); plot(1:T,cY,1:T,cM1,1:T,cM2); legend('raw data','rigid','non-rigid'); title('correlation coefficients','fontsize',14,'fontweight','bold')
%     subplot(2,3,5); scatter(cY,cM1); hold on; plot([0.9*min(cY),1.05*max(cM1)],[0.9*min(cY),1.05*max(cM1)],'--r'); axis square;
%         xlabel('raw data','fontsize',14,'fontweight','bold'); ylabel('rigid corrected','fontsize',14,'fontweight','bold');
%     subplot(2,3,6); scatter(cM1,cM2); hold on; plot([0.9*min(cY),1.05*max(cM1)],[0.9*min(cY),1.05*max(cM1)],'--r'); axis square;
%         xlabel('rigid corrected','fontsize',14,'fontweight','bold'); ylabel('non-rigid corrected','fontsize',14,'fontweight','bold');
%     linkaxes([ax1,ax2,ax3],'xy')
%     
% %% plot shifts        
% 
% shifts_r = horzcat(shifts1(:).shifts)';
% shifts_nr = cat(ndims(shifts2(1).shifts)+1,shifts2(:).shifts);
% shifts_nr = reshape(shifts_nr,[],ndims(Y)-1,T);
% shifts_x = squeeze(shifts_nr(:,1,:))';
% shifts_y = squeeze(shifts_nr(:,2,:))';
% 
% patch_id = 1:size(shifts_x,2);
% str = strtrim(cellstr(int2str(patch_id.')));
% str = cellfun(@(x) ['patch # ',x],str,'un',0);
% 
% figure;
%     ax1 = subplot(311); plot(1:T,cY,1:T,cM1,1:T,cM2); legend('raw data','rigid','non-rigid'); title('correlation coefficients','fontsize',14,'fontweight','bold')
%             set(gca,'Xtick',[])
%     ax2 = subplot(312); plot(shifts_x); hold on; plot(shifts_r(:,1),'--k','linewidth',2); title('displacements along x','fontsize',14,'fontweight','bold')
%             set(gca,'Xtick',[])
%     ax3 = subplot(313); plot(shifts_y); hold on; plot(shifts_r(:,2),'--k','linewidth',2); title('displacements along y','fontsize',14,'fontweight','bold')
%             xlabel('timestep','fontsize',14,'fontweight','bold')
%     linkaxes([ax1,ax2,ax3],'x')
% 
% %% plot a movie with the results
% 
% figure;
% for t = 1:1:T
%     subplot(121);
% %     imagesc(data.Y(:,:,t),[nnY,mmY]); xlabel('raw data','fontsize',14,'fontweight','bold'); axis equal; axis tight;
%     imagesc(data.Y(:,:,10,t),[nnY,mmY]); xlabel('raw data','fontsize',14,'fontweight','bold'); axis equal; axis tight;
% 
%     title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
%     subplot(122);imagesc(M1(:,:,10,t),[nnY,mmY]); xlabel('non-rigid corrected','fontsize',14,'fontweight','bold'); axis equal; axis tight;
%     title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
%     set(gca,'XTick',[],'YTick',[]);
%     drawnow;
%     pause(0.02);
% end
% 

%% ca_source_extraction.

% Test patches:


sizY = data.sizY;                       % size of data matrix
options = D.maskInfo.nmfoptions;

%

%K = 1705;
if D.maskInfo.params.patches
    
    fprintf('Processing patches for FILE %s (%i of %i tiffs).\n', filename, tiffidx, length(files));

    patches = construct_patches(sizY(1:end-1),patch_size,overlap);

    %% Run on patches (around 15 minutes)

    tic;
    [A,b,C,f,S,P,RESULTS,YrA] = run_CNMF_patches(data,K,patches,tau,p,options);
    fprintf('Completed CNMF patches for %i of %i tiffs.\n', tiffidx, length(files));

    results_fn = fullfile(D.nmfPath, sprintf('patch_results_File%03d_substack', tiffidx) );
    patch_results = matfile(results_fn, 'Writable', true);
    patch_results.RESULTS = RESULTS;
    patch_results.A = A;
    patch_results.b = b;
    patch_results.C = C;
    patch_results.f = f;
    patch_results.S = S;
    patch_results.P = P;
    patch_results.YrA = YrA;
    patch_results.patch_size = patch_size;
    patch_results.patch_overlap = overlap;

    fprintf('Saved results of run_CNMF_patches, File%03d.\n', tiffidx);

    toc

    % 256x256x20 volume:
    % Elapsed time is 2797.389282 seconds.


else

    [P,Y] = preprocess_data(data.Y,p);    

    %rois = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinalMask/analysis/datastruct_006/rois.mat');
    %roiA = rois.all;
    if D.maskInfo.seedRois && D.maskInfo.centroidsOnly
        P.ROI_list = double(D.maskInfo.seeds);
        fprintf('Starting with %i centroids as seeds.\n', length(P.ROI_list));
    end

    
    if ~usePreviousA

        if getref
            fprintf('Getting REF components! Specified ref tiff is %i.\n', D.maskInfo.ref.tiffidx);
        else
            fprintf('Told me to GETREF, and tiffidx %i is not the reference.\n', tiffidx)
            continue;
        end
        
        % Initialize components, or set input args:
        % -----------------------------------------------------------------
        if isfield(P, 'ROI_list')
            if D.maskInfo.centroidsOnly
                fprintf('Initializing components...\n');
                [Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize
            else
                fprintf('Using input masks (no initialization)...\n');
                Ain = D.maskInfo.seeds;
                Ain = Ain>0;
                Cin = [];
                fin = [];
                bin = [];
                fprintf('Size of input spatial footprint mask is: %s\n', mat2str(size(Ain)));
            end
        else
            fprintf('Initializing components...\n');
            [Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize
        end

        % Get centers of each ROI:
        % -----------------------------------------------------------------
        fprintf('Getting centroids...\n');
        centers = com(Ain,d1,d2,d3);
        if size(centers,2) == 2
            centers(:,3) = 1;
        end
        centers = round(centers);

        % Get avgs of each slice (instead of corr imgs):
        % -----------------------------------------------------------------
        fprintf('Averaging slices...\n');
        avgs = zeros([d1,d2,d3]);
        for slice=1:d3
            avgs(:,:,slice) = mean(Y(:,:,slice,:), 4);
            %avgs(:,:,slice) = mean(data.Y(:,:,slice,:), 4);
        end
        
        if show_plots
            plotCenteroverY(avgs, center, [d1,d2,d3]);  % plot found centers against max-projections of background image
        end
        
        % Update spatial components:
        % -----------------------------------------------------------------
        fprintf('Updating spatial...\n');
        Yr = reshape(Y,d,T);
        [A, b, Cin, P] = update_spatial_components(Yr, Cin, fin, [Ain, bin], P, options); 

       
        % Update temporal components:
        % -----------------------------------------------------------------
        fprintf('Size A, after update spatial: %s\n', mat2str(size(A)));
        fprintf('Updating temporal...\n');
        P.p = 0;
        [C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);
        % C is just Cin
        % YrA = AY - AA*C; % this is the same calc in extract_df_f for C2
        P.p = 2;
        
        if ~D.maskInfo.keepAll
            fprintf('Merging components...\n');
            fprintf('Starting size A: %s', mat2str(size(A))); 
            [Am, Cm, ~, ~, P] = merge_components(Yr, A, b, C, f, P, S, options); 
            [A, b, C, P] = update_spatial_components(Yr, Cm, f, [Am, b], P, options); 

            fprintf('Done merging. Post-merge size A is: %s', mat2str(size(A))); 
            P.p = p;
            fprintf('Updating temporal components again.\n');
            [C, F, P, S] = update_temporal_components(Yr, A, b, C, f, P, options);
        end        
        % Classify components:
        % -----------------------------------------------------------------
        %[ROIvars.rval_space,ROIvars.rval_time,ROIvars.max_pr,ROIvars.sizeA,ROIvars.keep] = classify_components(double(Y),A,C,b,f,YrA,options);
        %[A_or,C_or,S_or,P_or] = order_ROIs(A,C,S,P); % order components
        [~,background_df] = extract_DF_F(Yr,A,C,P,options);         
        %[C_df,~] = extract_DF_F(Yr,A,C,P,options); 
        
        % Extract fluorescence and DF/F on native temporal resolution:
        % -----------------------------------------------------------------
        % C is deconvolved activity, C + YrA is non-deconvolved fluorescence 
        % F_df is the DF/F computed on the non-deconvolved fluorescence
        
%         extractstart = tic();
%         
%         Ts = size(C,2);
%         tsub = options.tsub;
%         i = 1;
%         C_us = cell(1,1);    % cell array for thresholded fluorescence
%         f_us = cell(1,1);    % cell array for temporal background
%         P_us = cell(1,1);  
%         S_us = cell(1,1);
%         int = sum(floor(Ts(1:i-1)/tsub))+1:sum(floor(Ts(1:i)/tsub));
%         Cin_tmp = imresize([C(:,int);f(:,int)],[size(C,1)+size(f,1),Ts(i)]);
%         [C_us{i},f_us{i},P_us{i},S_us{i},YrA_us{i}] = update_temporal_components_fast(Yr,A,b,Cin_tmp(1:end-1,:),Cin_tmp(end,:),P,options);
%         size(C_us{i})
%         b_us{i} = max(mm_fun(f_us{i},Yr) - A*(C_us{i}*f_us{i}'),0)/norm(f_us{i})^2;
%         
%         prctfun = @(data) prctfilt(data,20,30);       % first detrend fluorescence (remove 20%th percentile on a rolling 1000 timestep window)
%         F_us = cellfun(@plus,C_us,YrA_us,'un',0);     % cell array for projected fluorescence
%         Fd_us = cellfun(prctfun,F_us,'un',0);         % detrended fluorescence
% 
%         Ab_d = cell(1,1);                            % now extract projected background fluorescence
%         Ab_d{i} = prctfilt((bsxfun(@times, A, 1./sum(A.^2))'*b_us{i})*f_us{i},20,30);
% 
%         F0 = cellfun(@plus, cellfun(@(x,y) x-y,F_us,Fd_us,'un',0), Ab_d,'un',0);   % add and get F0 fluorescence for each component
%         F_df = cellfun(@(x,y) x./y, Fd_us, F0 ,'un',0);                            % DF/F value
%         
%         fprintf('Extracted fluorescence traces!\n');
%         toc(extractstart);
        
    else
        
        % Load A from previous:
        % -----------------------------------------------------------------
        refnmf = matfile(D.maskInfo.ref.refnmfPath{1});
        refA = refnmf.A;
        Ain = refA>0;
        fprintf('size ref A: %s\n', mat2str(size(Ain)))
        refb = refnmf.b;
        bin = refb>0;
        fprintf('size ref b: %s\n', mat2str(size(bin)));
        fprintf('Using spatial comps from REF tiff %i, A islogical %i.\n', D.maskInfo.ref.tiffidx, islogical(Ain))
        
        % Update spatial components:
        % -----------------------------------------------------------------
        Yr = reshape(Y,d,T);
        fprintf('size Input A: %s\n', mat2str(size([Ain, bin])));
        %[A,b,Cin] = update_spatial_components(Yr, [],[], [Ain, bin], P, refnmf.options);
        [A,b,Cin] = update_spatial_components(Yr, [],[], Ain, P, refnmf.options);
 
        % Update temporal components:
        % -----------------------------------------------------------------
        P.p = 0;
        [C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,[],P,options);

        P.p = 2;
        
        % Classify components:
        % -----------------------------------------------------------------
        %[ROIvars.rval_space,ROIvars.rval_time,ROIvars.max_pr,ROIvars.sizeA,ROIvars.keep] = classify_components(double(Y),A,C,b,f,YrA,options);
        %[A_or,C_or,S_or,P_or] = order_ROIs(A,C,S,P); % order components
        [~,background_df] = extract_DF_F(Yr,A,C,P,options); 
        
        % Extract fluorescence and DF/F on native temporal resolution
        % -----------------------------------------------------------------
        % C is deconvolved activity, C + YrA is non-deconvolved fluorescence 
        % F_df is the DF/F computed on the non-deconvolved fluorescence
        extractstart = tic();
        Ts = size(C,2);
        tsub = options.tsub;
        i = 1;
        C_us = cell(1,1);    % cell array for thresholded fluorescence
        f_us = cell(1,1);    % cell array for temporal background
        P_us = cell(1,1);  
        S_us = cell(1,1);
        int = sum(floor(Ts(1:i-1)/tsub))+1:sum(floor(Ts(1:i)/tsub));
        Cin_tmp = imresize([C(:,int);f(:,int)],[size(C,1)+size(f,1),Ts(i)]);
        [C_us{i},f_us{i},P_us{i},S_us{i},YrA_us{i}] = update_temporal_components_fast(Y,A,b,Cin_tmp(1:end-1,:),Cin_tmp(end,:),P,options);
        b_us{i} = max(mm_fun(f_us{i},Yr) - A*(C_us{i}*f_us{i}'),0)/norm(f_us{i})^2;
        
        prctfun = @(data) prctfilt(data,20,30);       % first detrend fluorescence (remove 20%th percentile on a rolling 1000 timestep window)
        F_us = cellfun(@plus,C_us,YrA_us,'un',0);     % cell array for projected fluorescence
        Fd_us = cellfun(prctfun,F_us,'un',0);         % detrended fluorescence

        Ab_d = cell(1,1);                            % now extract projected background fluorescence
        Ab_d{i} = prctfilt((bsxfun(@times, A, 1./sum(A.^2))'*b_us{i})*f_us{i},20,30);

        F0 = cellfun(@plus, cellfun(@(x,y) x-y,F_us,Fd_us,'un',0), Ab_d,'un',0);   % add and get F0 fluorescence for each component
        F_df = cellfun(@(x,y) x./y, Fd_us, F0 ,'un',0);                            % DF/F value
        fprintf('Extracted fluorescence traces!\n');
        toc(extractstart);
        
    end
       

end

%% compute correlation image on a small sample of the data (optional - for visualization purposes) 

% Cn = correlation_image_max(single(data.Y),8);
% 
Cn = correlation_image_3D(data.Y); 
% 
%     
% %% classify components
% 
% [ROIvars.rval_space,ROIvars.rval_time,ROIvars.max_pr,ROIvars.sizeA,keep] = classify_components(data,A,C,b,f,YrA,options);
% 
% 
% %% display centers of found components
% if show_plots
%     plotCenteroverY(Cn, center, [d1,d2,d3]);  % plot found centers against max-projections of background image
% end

%% Plot components and view traces:

% Cn looks crappy, try just avg to do sanity check of ROIs:
avgs = zeros([d1,d2,d3]);
for slice=1:d3
    avgs(:,:,slice) = mean(data.Y(:,:,slice,:), 4);
end

%[T_out, Y_r_out, C_out, Df_out] = plot_components_3D_GUI(data.Y,A,C,b,f,Cn,options);
%[T_out, Y_r_out, C_out, Df_out] = plot_components_3D_GUI(data.Y,A,C,b,f,avgs,options);
% [T_out, Y_r_out, C_out, Df_out] = plot_components_3D_GUI(Y,A,C,b,f,avgs,options);
if show_plots
    plot_components_3D_GUI(Y,A,C,b,f,avgs,options);
end

%[T_out, Y_r_out, C_out, Df_out] = plot_components_3D_GUI(data.Y,patch.A,patch.C,patch.b,patch.f,avgs,options);

%% SAVE nmf output:

if getref
    nmf_outfile = ['nmfoutput_ref_' filename, '.mat']
else
    nmf_outfile = ['nmfoutput_' filename, '.mat']
end
        
        
nmf_outputpath = fullfile(D.nmfPath, nmf_outfile);
nmfoutput = matfile(nmf_outputpath, 'Writable', true);

nmfoutput.mempath = mempath;
nmfoutput.outpath = nmf_outputpath; %fullfile(sourcepath, savedir);
nmfoutput.tiff = [filename, '.tif'];
nmfoutput.K = K;
nmfoutput.tau = tau;
nmfoutput.merge_thr = merge_thr;
nmfoutput.p = p;
nmfoutput.options = options;
nmfoutput.motion = MC;

nmfoutput.Cn = Cn;
nmfoutput.avgs = avgs;

nmfoutput.A = A;
nmfoutput.P = P;
nmfoutput.S = S;
nmfoutput.C = C;
nmfoutput.b = b;
nmfoutput.f = f;
nmfoutput.YrA = YrA; % add C to get non-deconvolved fluorescence
nmfoutput.Y = data.Y;
if ~getref
    nmfoutput.background_df = background_df;    % Divide C by this to get "inferred"
    %nmfoutput.classify = ROIvars;               % output of classify_components
    nmfoutput.Fd_us = Fd_us;                    % deterended fluorescence -- C + YrA (non-deconv fluorescence)
    nmfoutput.F0 = F0;                          % background for each component
    nmfoutput.F_df = F_df;                      % % DF/F value
end

%nmfoutput.center = center;

fprintf('Finished CNMF trace extraction for %i of %i files.\n', tiffidx, length(files));
fprintf('TOTAL ELAPSED TIME for file %i: \n')
toc(first_tic);

nmf_outpaths{end+1} = nmf_outputpath; %fullfile(maskPath, nmfStructName);

close all;

end
