function outmap = assign_roimap(maskcell, roimap, roivecs, varargin)

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
            idx = find(freqs==freqs(roivecs(:,i)==max(roivecs(:,i))))
            if length(idx)>1
                idx = idx(1);
            end
            outmap(maskcell{i}) = roimap(idx,i);
        end
    case 0
        % simple case:  assign ROI in 'roimap' to corresponding roi idx
        % provided in 'roivecs'
        for i=1:length(maskcell)
            roimap(maskcell{i}) = roivecs(i);
        end
        outmap = roimap;
end
            
    
end
