function deinterleave_tiffs(newtiff, filename, acquisition_name, nslices, nchannels, fid,  write_dir)

namingFunction = @defaultNamingFunction;

fprintf('Saving deinterleaved slices to:\n%s\n', write_dir);
for sl = 1:nslices
        
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
        mov_filename = feval(namingFunction,acquisition_name, sl, ch, fid);
        try
            tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
            tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
        catch
            % Sometimes, disk access fails due to intermittent
            % network problem. In that case, wait and re-try once:
            pause(60);
            tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
        end
        
    else
        for ch = 1:nchannels
            
            frame_idx = ch + (sl-1)*nchannels;
            
            % Create movie fileName and save to default format
            % TODO: set this to work with other mc methods....
            mov_filename = feval(namingFunction,acquisition_name, sl, ch, fid);
            try
                tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir);
            catch
                % Sometimes, disk access fails due to intermittent
                % network problem. In that case, wait and re-try once:
                pause(60);
                tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir);
            end
        end
    end
end

end

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end


