% Get centroids of cNMF ROIs:

nmf = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/nmf/nmfoutput_File004_substack.mat')
masks = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/masks/nmf3D_masks_File004.mat')
expressingcells = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/expressingCells.mat')
expressingcells = expressingcells.cellIDs;

A = nmf.A;
d1 = 120;
d2 = 120;
d3 = 22;
cm = com(A,d1,d2,d3);
centers = cm';

cEM = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids/centroids_EM.mat');
cellnames = fieldnames(cEM);
cell_index = [];
for c = 1:length(expressingcells)
    foundcell = sprintf('cell%04d', expressingcells(c));
    for nameidx=1:length(cellnames)
        if strcmp(foundcell, cellnames{nameidx})
            cell_index = [cell_index nameidx];
        end
    end
end


expressing_centroids = zeros(length(expressingcells), 3);
expressing_centroids_float = zeros(length(expressingcells), 3);
for i=1:length(cell_index)
    expressing_centroids(i,:) = masks.centers(cell_index(i),:);
    expressing_centroids_float(i,:) = centers(cell_index(i),:);
end

save_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/em7_centroids';
expressing_cells_info = 'expressing_cells_name_idx.mat';
cells = struct();
cells.cellIDs = expressingcells;
cells.cell_index = cell_index;
cells.found_centroids = expressing_centroids_float;
cells.found_centroids_pix = expressing_centroids;
save(fullfile(save_path, expressing_cells_info), '-struct', 'cells');

