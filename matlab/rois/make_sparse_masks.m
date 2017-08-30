function spMask = make_sparse_masks(maskarray)
        
    [i,j,s] = find(maskarray);
    [m,n] = size(maskarray);
    spMask = sparse(i,j,s,m,n); %(:,:,roi);

end
