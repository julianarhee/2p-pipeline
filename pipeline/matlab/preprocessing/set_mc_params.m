function options = set_mc_params(varargin)

% run_multi_acquisitions = false;                  % TODO: Set this true if processing  multiple Acquisitions (i.e., FOVs)
% mcparams.crossref = false;                       % TODO: True if same acquisition, but different stimuli/experiments that should be co-registered (for ex., retinotopy and RSVP stimuli)
% 
% mcparams.corrected = true;
% mcparams.method = 'Acquisition2P';                 % Source for doing correction. Can be custom. ['Acqusition2P', 'NoRMCorre']
% mcparams.flyback_corrected = true;                 % True if did correct_flyback.py 
% 
% mcparams.ref_channel = 1;                          % Ch to use as reference for correction
% mcparams.ref_file = 3;                             % File index (of numerically-ordered TIFFs) to use as reference
% mcparams.algorithm = @lucasKanade_plus_nonrigid;   % Depends on method/source: - Acq_2P [@lucasKanade_plus_nonrigid, @withinFile_withinFrame_lucasKanade];
%                                                    %                           - NoRMCorre ['rigid', 'nonrigid']
% 
% mcparams.split_channels = true;                    % MC methods should parse corrected-tiffs by Channel-File-Slice (Acq2P does this already). Last step interleaves parsed tiffs, but sometimes they are too big for Matlab
% mcparams.bidi_corrected = true;                    % For faster scanning, SI option for bidirectional-scanning is True -- sometimes need extra scan-phase correction for this
% 

Names = [
    'corrected          '       % corrected or raw (T/F)
    'method             '       % Source for doing correction. Can be custom. ['Acqusition2P', 'NoRMCorre']
    'flyback_corrected  '       % True if did correct_flyback.py 
    'ref_channel        '       % Ch to use as reference for correction
    'ref_file           '       % File index (of numerically-ordered TIFFs) to use as reference
    'algorithm          '       % Depends on 'method': Acq_2P [@lucasKanade_plus_nonrigid, @withinFile_withinFrame_lucasKanade], NoRMCorre ['rigid', 'nonrigid']
    'split_channels     '       % *MC methods should parse corrected-tiffs by Channel-File-Slice (Acq2P does this already). Last step interleaves parsed tiffs, but sometimes they are too big for Matlab
    'bidi_corrected     '       % *For faster scanning, SI option for bidirectional-scanning is True -- sometimes need extra scan-phase correction for this
    'source_dir         '       % Path to folder containing TIFFs to be corrected (A.data_dir)
    'crossref           '       % True if correcting across experiment-types (but same FOV). (Not fully tested)
    'nchannels          '       % From ref-struct. useful here, for post-mc-cleanup steps. [default: 1]
    'dest_dir           '
    'info               '
    ]; 
    %'bidi_corrected_dir '
    %'parsed_dir         '
    %'info               '    
    %];

% TODO: * fields indicate potential auto-populated fields based on
% evaluation metrics

% TODO:  incorporate try/catch for splitting channels
% TODO:  include MC-evaluation step for extra BiDi correction

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
    {true}                                      % corrected
    {'Acquisition2P'}                           % method
    {false}                                     % flyback_corrected
    {1}                                         % ref_channel
    {1}                                         % ref_file
    {@withinFile_withinFrame_lucasKanade}       % algorithm
    {false}                                     % split_channels
    {true}                                      % bidi_corrected
    {''}                                        % tiff_dir
    {false}                                     % crossref
    {1}                                         % nchannels
    {'Corrected'}                               % corrected_dir
    {struct()}                                  % info struct
    ];
% 
%     {'Corrected_Bidi'}                          % bidi_corrected_dir
%     {'Parsed'}                                  % parsed_dir
%     {struct()}                                  % info (struct)
%     ];
% 
for j = 1:m
    if eval(['isempty(options.' Names(j,:) ')'])
        eval(['options.' Names(j,:) '= Values{j};']);
    end
end

end
