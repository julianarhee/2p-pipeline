function D = set_mask_paths(D, dsoptions)

% --------------------------------------------------------------------
% Specify sources for input masks, if they are being provided.
% --------------------------------------------------------------------
% NOTES:
% A little convoluted, but want to maintain uniform path-accessing.
% TODO:  may want to make this less complicated, but currently, works
% with file-organization and how analyses are accessed (including GUI)
% --------------------------------------------------------------------

% Get path to masks
if isempty(dsoptions.roipath)
    [maskName, maskSource, ~] = uigetfile;
else
    [maskSource, maskName, ~] = fileparts(dsoptions.maskpath);
end
refidx = dsoptions.maskrefidx;

roiType = dsoptions.roitype; %'3Dcnmf' 
maskDimensions = dsoptions.maskdims; %'3D'; 

seedRois = dsoptions.seedrois; %true;
manual3Dshape = dsoptions.maskshape; %'3Dcontours' %'spheres'
maskFinder = dsoptions.maskfinder; %'centroids'

% 2.b.  Specify additional args, depending on ROI type:
switch maskDimensions
    case '2D'
        switch roiType
            case 'manual2D'
                [fpath,fcond,~] = fileparts(D.sourceDir);
                D.maskSource = fcond;

            case 'condition'
                % Test cross-ref:

                D.maskSourcePath = maskSource; %refMaskStruct; %'retinotopy1';
                D.maskDidx = refidx; %refMaskStructIdx;
                D.maskDatastruct = sprintf('datastruct_%03d', D.maskDidx);
                fprintf('Using pre-defined masks from condition: %s.\n', D.maskSource);

                pathToMasks = fullfile(D.maskSourcePath, 'analysis', D.maskDatastruct, 'masks');
                maskNames = dir(fullfile(pathToMasks, 'masks_*'));
                maskNames = {maskNames(:).name}';
                D.maskPaths = cell(1, length(maskNames));
                for maskIdx=1:length(maskNames)
                    D.maskPaths{maskIdx} = fullfile(pathToMasks, maskNames{maskIdx});
                end

            case 'pixels'
                [fpath,fcond,~] = fileparts(D.sourceDir);
                D.maskSource = fcond;

            case 'roiMap'
                % roiCentroidPaths = dir(fullfile(maskSource, centroidfnpattern));
                % if length(roiCentroidPaths)==1
                %     roiPath = roiCentroidPaths(1).name;
                % else
                %     roiIdx = 1; % Specify which ROI maps to use, if more than 1
                %     roiPath = roiCentroidPaths(roiIdx).name;
                % end
                D.maskSource = dsoptions.maskpath; %mapSource;

            case 'cnmf'
                [fpath,fcond,~] = fileparts(D.sourceDir);
                D.maskSource = fcond;
        end

    case '3D'
        fprintf('creating 3D masks...\n');

        D.maskType = manual3Dshape;

        if seedRois
            fprintf('seeding rois...\n')
            % roiCentroidPaths = dir(fullfile(mapSource, centroidfnpattern));         % Path to .mat containg centroid info
            % roiMaskPaths = dir(fullfile(mapSource, maskfnpattern));

            % if length(roiCentroidPaths)==1
            %     roiCentroidPath = roiCentroidPaths(1).name;
            % else
            %     roiIdx = 1; % Specify which ROI maps to use, if more than 1
            %     roiCentroidPath = roiCentroidPaths(roiIdx).name;
            % end
            % if length(roiMaskPaths)==1
            %     roiMaskPath = roiMaskPaths(1).name;
            % else
            %     roiIdx = 1; % Specify which ROI maps to use, if more than 1
            %     roiMaskPath = roiMaskPaths(roiIdx).name;
            % end
            D.maskSource = dsoptions.maskpath; %mapSource;
        else
            [fpath,fcond,~] = fileparts(D.sourceDir);
            D.maskSource = fcond;
        end
end
D.maskSource = dsoptions.maskpath;
D.seedRois = seedRois;
D.roiType = roiType;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('Updated datastruct analysis: %s\n', D.datastructPath)

end
