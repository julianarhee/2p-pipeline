#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 12:40:34 2018

@author: juliana
"""


# PCA:  reduce zscores projected onto normals
# -----------------------------------------------------------------------------
from sklearn.decomposition import PCA

svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
svc.fit(cX_std, cy)
#
#if len(svc.classes_) > 2:
#    cdata = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))])
#else:
#    cdata = np.array(svc.coef_[0].dot(cX_std.T) + svc.intercept_[0])
##    
#pX = cdata.T # (n_samples, n_features)
pca_data = 'inputdata' #'projdata' # 'inputdata'

if pca_data == 'inputdata':
    pX = cX_std.copy()
    py = cy.copy()

elif pca_data == 'projdata':
    pX = cdata #copy()
    py = cy.copy() #svc.classes_
    
ncomps = 2 #pX.shape[-1]
print "N comps:", ncomps


pca = PCA(n_components=ncomps)
pca.fit(pX)

principal_components = pca.fit_transform(pX)

pca_df = pd.DataFrame(data=principal_components,
                      columns=['pc%i' % int(i+1) for i in range(ncomps)])
pca_df.shape

labels_df = pd.DataFrame(data=py,
                         columns=['target'])

pdf = pd.concat([pca_df, labels_df], axis=1).reset_index()

stimtype = sconfigs['config001']['stimtype']
if class_name == 'ori':
    curr_colors = sns.color_palette("hls", len(class_labels))
else:
    curr_colors = sns.color_palette("RdBu_r", len(class_labels))

    
#colors = ['r', 'orange', 'y', 'g', 'b', 'c', 'm', 'k']

# Visualize 2D projection:
fig = pl.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA-reduced k-D neural state for each tpoint', fontsize = 20)
#for target, color in zip(orientations,colors):
for target, color in zip(class_labels, curr_colors):

    indicesToKeep = pdf['target'] == target
    ax.scatter(pdf.loc[indicesToKeep, 'pc1'],
               pdf.loc[indicesToKeep, 'pc2'],
               c = color,
               s = 50,
               alpha=0.5)
ax.legend(class_labels)
ax.grid()

pl.title('PCA, %s (%s)' % (pca_data, data_type))

if pca_data == 'inputdata':
    figname = 'PCA_%s_%s.png' % (pca_data, data_type)
    pl.savefig(os.path.join(clf_basedir, figname))
elif pca_data == 'projdata':
    figname = '%s__PCA_%s.png' % (classif_identifier, pca_data)
    pl.savefig(os.path.join(population_figdir, figname))


#%%

# try t-SNE:
# -----------------------------------------------------------------------------

from sklearn.manifold import TSNE
import time

#pX = cdata.T # (n_samples, n_features)
pca_data ='inputdata' # 'inputdata'

if pca_data == 'inputdata':
    pX = cX_std.copy()
    py = cy.copy()

elif pca_data == 'projdata':
    pX = np.array([svc.coef_[c].dot(cX_std.T) + svc.intercept_[c] for c in range(len(svc.classes_))]).T

    py = cy.copy() #svc.classes_
    
print pX.shape


if pca_data == 'inputdata':
    tsne_df = pd.DataFrame(data=pX,
                   index=np.arange(0, pX.shape[0]),
                   columns=['r%i' % i for i in range(pX.shape[1])]) #(ori) for ori in orientations])
    
elif pca_data == 'projdata':
    # Reduce PROJECTED data
    tsne_df = pd.DataFrame(data=pX,
                       index=np.arange(0, pX.shape[0]),
                       columns=[str(label) for label in class_labels])

feat_cols = [f for f in tsne_df.columns]

# Visualize:
#target_ids = range(len(digits.target_names))
target_ids = range(len(class_labels))

multi_run = True
nruns = 4

if multi_run is False:
    nruns = 1

perplexity = 40 #100# 40 #100 #5# 100
niter = 3000 #5000

colors = curr_colors # 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'

if multi_run:
    fig, axes = pl.subplots(2, nruns/2, figsize=(12,8))
    axes = axes.ravel().tolist()
    for run in range(nruns):

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=niter)
        tsne_results = tsne.fit_transform(tsne_df[feat_cols].values)
        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

        ax = axes[run]
        print run
        for i, c, label in zip(target_ids, colors, class_labels):
            ax.scatter(tsne_results[py == int(label), 0], tsne_results[py == int(label), 1], c=c, label=label, alpha=0.5)
            box = ax.get_position()
            ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                             box.width * 0.98, box.height * 0.98])

else:

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=niter)
    tsne_results = tsne.fit_transform(tsne_df[feat_cols].values)
    print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

    fig, ax = pl.subplots(1, figsize=(6, 6))
    colors = curr_colors # 'r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k' #, 'purple'
    #for i, c, label in zip(target_ids, colors, digits.target_names):
    #    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    for i, c, label in zip(target_ids, colors, class_labels):
        pl.scatter(tsne_results[cy == int(label), 0], tsne_results[cy == int(label), 1], c=c, label=label)
        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.01, box.y0 + box.height * 0.02,
                         box.width * 0.98, box.height * 0.98])

# Put a legend below current axis
pl.legend(loc=9, bbox_to_anchor=(-0.2, -0.15), ncol=len(class_labels))

if pca_data == 'inputdata':
    pl.suptitle('t-SNE, proj %s (%i-D rois) | px: %i, ni: %i' % (data_type, nrois, perplexity, niter))
    figname = 'tSNE_%s_%irois_orderedT_pplex%i_niter%i_%iruns.png' % (data_type, nrois, perplexity, niter, nruns)
    pl.savefig(os.path.join(clf_basedir, figname))

elif pca_data == 'projdata':
    figname = 'tSNE_proj_onto_norm_%s_pplex%i_niter%i_%iruns_%s.png' % (data_type, perplexity, niter, nruns, classif_identifier)

    if data_type == 'xcondsub':
        pl.suptitle('t-SNE, proj norm (xcondsub time-series) | px: %i, ni: %i' %  (perplexity, niter))
    else:
        pl.suptitle('t-SNE: proj norm (%s) | px: %i, ni: %i' %  (data_type, perplexity, niter))
    pl.savefig(os.path.join(population_figdir, figname))
