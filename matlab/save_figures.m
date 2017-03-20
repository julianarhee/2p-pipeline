function save_figures(fig, figPath, figName)
    
    if ~exist(figPath, 'dir')
        mkdir(figPath);
    end
    figTypes = {'.fig', '.png', '.pdf'};
    for iidx=1:length(figTypes)
        
        figNameAppend = strcat(figName, figTypes{iidx})
        saveas(fig, fullfile(figPath, figNameAppend));

    end
    
    fprintf('Saved figures to path: %s\n', figPath);
    fprintf('Figure base name is: %s\n', figName);

    close(fig)
end