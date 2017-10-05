function options = DFoptions(varargin)

% This is lifted from CNMFSetParams.m as a way to provide large set of inputs without things being overly clunky...

% TODO:  create txt file with datastruct info.
%dstructFile = 'datastructs.txt'
%dstructPath = fullfile(source, session, run, dstructFile)
%headers = {'datastruct', 'acquisition', 'tefo', 'preprocessing', 'meta',...
%            'roiType', 'seedrois', 'maskdims', 'maskshape', 'maskfinder',...
%            'channels', 'signalchannel', 'slices', 'averaged',...
%             'excludedtiffs', 'metaonly', 'hugetiffs'};
%
%fid = fopen(dstructPath, 'w');
%if ~exist(dstructPath)
%    fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\r\n', headers{1:end});
%else
%    fprintf(fid, '%s\t%s\t%i\t%s\%s\t%s\t%i\t%s\t%s\t%s\t%i\t%i\t%s\t%i\t%s\t%s\t%i\t%i', dstructoptions{1:end});
%    fclose(fid);
%end
%

Names = [
    % datastruct info
    'source             '       % source (parent) directory
    'session            '       % session name (single FOV)
    'run                '       % experiment type (e.g., 'retinotopyX')
    'datastruct         '       % datastruct index (default: 1)
    'acquisition        '       % acquisition name
    'datapath           '       % alternate datapath if preprocessed (default: '')
    % acquisition info
    'tefo               '       % microscope type/revision (default: false)
    'preprocessing      '       % tiff processing ['raw', 'fiji', 'Acquisition2P']
    'corrected          '       % motion corrected, standard is Acquisition2P methods (default: true)
    'meta               '       % source of meta info ['SI', 'manual'] (default: 'SI')
    'channels           '       % number of acquired channels (default: 2)
    'signalchannel      '       % channel num of activity channel (default: 1)
    % ROI mask info
    'roitype            '       % roi mask type ['manual2D', 'pixels', 'roiMap', 'cnmf', 'manual3Drois', '3Dcnmf']
    'seedrois           '       % seeds (centroids or masks) for initial ROI locs (default: false)
    'maskpath           '       % path to file of masks/centroids/etc (must specify if 'seedrois')
    'maskdims           '       % dimensions of ROI masks ['2D', '3D']
    'maskshape          '       % mask shape ['circles','contours','3Dcontours','spheres']
    'maskfinder         '       % mask source, if providing ['blobDetector', 'EMmasks', 'centroids'] 
    % analysis meta info
    'slices             '       % indices of acquired slices that are included for analysis (data-only)
    'averaged           '       % single runs/trials or averaged (default: false)
    'matchedtiffs       '       % array of tiff idxs to average together (default: [[]])
    'excludedtiffs      '       % acquired tiffs that should be excluded from analysis (default: [])
    'metaonly           '       % only get meta info because tiff files too large (default: false)
    'nmetatiffs         '       % number of tiffs need meta-only info for (default: 0)
    'memmapped          '       % use memmapped or no
    'correctbidi        '       % correct bidirectional scan phase offset
    'reference          '       % reference File for motion-correction, averaged slice views, ROI reference, etc.
    'stimulus           '       % stimulus type ['bar', 'grating', 'image']
    ];



[m, n] = size(Names);
names = lower(Names);

options = [];
for nameidx=1:m
    eval(['options.' Names(nameidx,:) ' = [];']);
end

i = 1;
while i <= nargin
    arg = varargin{i};
    if ischar(arg), break; end
    if ~isempty(arg)
        if ~isa(arg,'struct')
            error(sprintf(['Expected argument %d to be a string parameter name ' ...
                'or an options structure\ncreated with OPTIMSET.'], i));
        end
        for j = 1:m
            if any(strcmp(fieldnames(arg),deblank(Names(j,:))))
                eval(['val = arg.' Names(j,:) ';']);
            else
                val = [];
            end
            if ~isempty(val)
                eval(['options.' Names(j,:) '= val;']);
            end
        end
    end
    i = i + 1;
end

if rem(nargin-i+1,2) ~= 0
    error('Arguments must occur in name-value pairs.');
end

expectval = 0;                          % start expecting a name, not a value
while i <= nargin
    arg = varargin{i};
    
    if ~expectval
        if ~ischar(arg)
            error(sprintf('Expected argument %d to be a string parameter name.', i));
        end
        
        lowArg = lower(arg);
        j = strmatch(lowArg,names);
        if isempty(j)                       % if no matches
            error(sprintf('Unrecognized parameter name ''%s''.', arg));
        elseif length(j) > 1                % if more than one match
            % Check for any exact matches (in case any names are subsets of others)
            k = strmatch(lowArg,names,'exact');
            if length(k) == 1
                j = k;
            else
                msg = sprintf('Ambiguous parameter name ''%s'' ', arg);
                msg = [msg '(' deblank(Names(j(1),:))];
                for k = j(2:length(j))'
                    msg = [msg ', ' deblank(Names(k,:))];
                end
                msg = sprintf('%s).', msg);
                error(msg);
            end
        end
        expectval = 1;                      % we expect a value next
        
    else
        eval(['options.' Names(j,:) '= arg;']);
        expectval = 0;
        
    end
    i = i + 1;
end

if expectval
    error(sprintf('Expected value for parameter ''%s''.', arg));
end



Values = [
    % datastruct info
    {'source'}                  % 'source             '       % source (parent) directory
    {'session'}                  % 'session            '       % session name (single FOV)
    {'run'}                  % 'run                '       % experiment type (e.g., 'retinotopyX')
    {1}                  % 'datastruct         '       % datastruct index (default: 1)
    {'acquisition'}                  % 'acquisition        '       % acquisition name
    {''}                             % 'datapath           '       % alternate datapath of preprocessed
    % acquisition tiff info
    {false}                  % 'tefo               '       % microscope type/revision (default: false)
    {'raw'}                  % 'preprocessing      '       % tiff processing ['raw', 'fiji', 'Acquisition2P']
    {false}                  % 'corrected          '        % motion-corrected (with Acq2P)
    {'SI'}                  % 'meta               '       % source of meta info ['SI', 'manual'] (default: 'SI')
    {2}                  % 'channels           '       % number of acquired channels (default: 2)
    {1}                  % 'signalchannel      '       % channel num of activity channel (default: 1)
    % ROI mask info
    {'pixels'}                  % 'roitype            '       % type of roi mask ['manual2D', 'pixels', 'roiMap', 'cnmf', 'manual3Drois', '3Dcnmf']
    {false}                  % 'seedrois           '       % provide seeds (centroids or masks) for initial ROI locations (default: false)
    {''}		     % 'maskpath'	
    {'2D'}                  % 'maskdims           '       % dimensions of ROI masks ['2D', '3D']
    {'circles'}                  % 'maskshape          '       % mask shape ['circles','contours','3Dcontours','spheres']
    {''}                  % 'maskfinder         '       % source of ROIs, if providing masks ['blobDetector', 'EMmasks', 'centroids'] (default: '')
    % analysis meta info
    {mat2str([1:20])}                  % 'slices             '       % indices of acquired slices that are included for analysis (data-only, default: [1:20])
    {false}                  % 'averaged           '       % single runs/trials or averaged (default: false)
    {[]}                     % 'matchedtiffs       '       % average matching tiffs or no?
    {[]}                  % 'excludedtiffs      '       % acquired tiffs that should be excluded from analysis (default: [])
    {false}                  % 'metaonly           '       % only get meta info because tiff files too large (default: false)
    {0}                  % 'hugetiffs          '       % number of tiffs need meta-only info for (default: 0)
    {true}              % 'memmapped or no for get3DRois...' TMP
    {false}
    {1}
    {''}    
    ];

for j = 1:m
    if eval(['isempty(options.' Names(j,:) ')'])
        eval(['options.' Names(j,:) '= Values{j};']);
    end
end

end
