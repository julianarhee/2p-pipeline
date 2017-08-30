function spMask = makeSparseMasks(maskarray)
        
    [i,j,s] = find(maskarray);
    [m,n] = size(maskarray);
    spMask = sparse(i,j,s,m,n); %(:,:,roi);

end