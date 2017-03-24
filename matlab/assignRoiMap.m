function outmap = assignRoiMap(maskcell, roimap, roivecs, varargin)

nargin = length(varargin);

switch nargin
    case 1
        idx = varargin{1};
        for i=1:length(maskcell)
            roimap(full(maskcell{i})==1) = roivecs(i, idx);
        end
        outmap = roimap;
    case 2
        %sourcemat = varargin{2};
        freqs = varargin{2};
        outmap = zeros(size(maskcell{1},1), size(maskcell{1},2));
        for i=1:length(maskcell)
            idx = find(freqs==freqs(roivecs(i,:)==max(roivecs(i,:))));
            outmap(full(maskcell{i})==1) = roimap(i, idx);
        end
    case 0
        for i=1:length(maskcell)
            roimap(full(maskcell{i})==1) = roivecs(i);
        end
        outmap = roimap;
end
            
    
end