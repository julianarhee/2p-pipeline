function deinterleave_tiffs(newtiff, filename, fid, write_dir, A)

namingFunction = @defaultNamingFunction;

base_filename = A.base_filename;
%nslices = length(I.slices);
%slices = I.slices;
slices = A.slices;
nslices = length(A.slices);
nchannels = A.nchannels;

fprintf('Saving deinterleaved slices to:\n%s\n', write_dir);
for sl = slices %1:nslices
        
    % Check filename to see if Ch1/Ch2 split:
    if strfind(filename, 'Channel01')
        ch=1;
        channels_are_split = true;
    elseif strfind(filename, 'Channel02')
        ch=2;
        channels_are_split = true;
    else
        channels_are_split = false;
    end

    if channels_are_split
         
        frame_idx = sl; %1 + (sl-1)*nchannels;

        % Create movie fileName and save to default format
        % TODO: set this to work with other mc methods....
        mov_filename = feval(namingFunction, base_filename, sl, ch, fid);
        try
            tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir, 'int16');
        catch
            % Sometimes, disk access fails due to intermittent
            % network problem. In that case, wait and re-try once:
            pause(60);
            tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir, 'int16');
        end
        
    else
        for ch = 1:nchannels
            
            frame_idx = ch + (sl-1)*nchannels;
            
            % Create movie fileName and save to default format
            % TODO: set this to work with other mc methods....
            mov_filename = feval(namingFunction, base_filename, sl, ch, fid);
            try
                tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir, 'int16');
            catch
                % Sometimes, disk access fails due to intermittent
                % network problem. In that case, wait and re-try once:
                pause(60);
                tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir, 'int16');
            end
        end
    end
    fprintf('Finished splicing Slice %i, Channel %i.\n', sl, ch);
end

end

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end


