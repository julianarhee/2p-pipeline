function outmap = assign_roimap_3D(maskcell, centers, nslices, roimap, roivecs, varargin)

nargin = length(varargin);

switch nargin
    case 1
        % standard case: assign ROI in 'roimap' to value at 'idx' (index
        % into 'roivecs') 
        idx = varargin{1};
        for i=1:length(maskcell)
            roimap(maskcell{i}) = roivecs(idx,i);
        end
        outmap = roimap;
    case 2
        % max case:  find 'idx' that meets some condition related to
        % 'freqs', then use this to index into 'roimap'
        freqs = varargin{2};
        outmap = zeros(size(maskcell{1},1), size(maskcell{1},2));
        for i=1:length(maskcell)
            idx = find(freqs==freqs(roivecs(:,i)==max(roivecs(:,i))));
            outmap(maskcell{i}) = roimap(idx,i);
        end
    case 0
        % simple case:  assign ROI in 'roimap' to corresponding roi idx
        % provided in 'roivecs'
        mcell = cell(1, nslices);
        for mc=1:length(mcell)
            smap = zeros(size(roimap));
            for i=1:length(maskcell)
                if mc==centers(i,3)
                    smap(maskcell{i}(:,:,centers(i,3))) = roivecs(i);
                end
            end
            mcell{mc} = smap;
        end
        
        outmap = cat(3,mcell{1:end});
end
            
    
end
