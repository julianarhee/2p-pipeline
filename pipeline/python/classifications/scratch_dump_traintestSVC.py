#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:31:10 2018

@author: juliana
"""


        
#%%
# #############################################################################
# Plot grand-mean traces pair-wise between conditions:
# #############################################################################
        
        
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

#fig, axes = pl.subplots(n_classes, ntrials_per_stim, sharex=True, sharey=True, figsize=(20,20))
config = 'config006'
nframes_in_trial = max([all_preds[config].shape[0] for config in all_preds.keys()])

nclasses_total = all_preds[config].shape[1]
ntrials_curr_config = all_preds[config].shape[2]

configs_tested = sorted(list(set(labels_df['config'])), key=natural_keys)

    
# Figure settings:
nr = len(configs_tested)
nc = len(configs_tested)
aspect = 1
bottom = 0.1; left=0.1
top=1.-bottom; right = 1.-left
fisasp = 1 # (1-bottom-(1-top))/float( 1-left-(1-right) )
wspace=0.15  # widthspace, relative to subplot size; set to zero for no spacing
hspace=wspace/float(aspect)
figheight= 8 # fix the figure height
figwidth = (nc + (nc-1)*wspace)/float((nr+(nr-1)*hspace)*aspect)*figheight*fisasp



# Colormap settings:
tag = np.arange(0, nframes_in_trial)
cmap = pl.cm.jet                                        # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]             # extract all colors from the .jet map
#cmaplist[0] = (.5,.5,.5,1.0)                            # force the first color entry to be grey
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
bounds = [int(i) for i in np.linspace(0,len(tag),len(tag)+1)]             # define the bins and normalize
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)




fig, axes = pl.subplots(nr, nc, sharex=True, sharey=True, figsize=(figwidth, figheight))
pl.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                    wspace=wspace, hspace=hspace)


plot_configs = sorted(configs_tested, key=lambda x: (test_configs[x]['stim_dur'], test_configs[x]['ori'], test_configs[x]['direction']))
#plot_configs = svc.classes_

for row in range(nr):
    row_config = plot_configs[row]
    for col in range(nc):
        col_config = plot_configs[col]
        row_label = '%i_%s_%i' % (test_configs[row_config]['ori'], test_configs[row_config]['direction'], test_configs[row_config]['stim_dur'])
        col_label = '%i_%s_%i' % (test_configs[col_config]['ori'], test_configs[col_config]['direction'], test_configs[col_config]['stim_dur'])
        ax = axes[row][col]
        
#        traceX = np.mean(np.mean(test_traces[row_config], axis=2), axis=1)
#        traceY = np.mean(np.mean(test_traces[col_config], axis=2), axis=1)
#        traceX = svc.coef_[row, :].dot(np.mean(test_traces[row_config], axis=2).T) + svc.intercept_[row]
#        traceY = svc.coef_[col, :].dot(np.mean(test_traces[col_config], axis=2).T) + svc.intercept_[col]
        traceX = np.mean(test_traces[row_config], axis=1)
        traceY = np.mean(test_traces[col_config], axis=1)

        for rep in range(traceX.shape[1]):
            currx = traceX[:, rep]
            curry = traceY[:, rep]
            
            if currx.shape[0] > curry.shape[0]:
                curry = np.pad(curry, (0, currx.shape[0]-curry.shape[0]), mode='constant', constant_values=np.nan)
            elif traceY.shape[0] > traceX.shape[0]:
                currx = np.pad(currx, (0, curry.shape[0]-currx.shape[0]), mode='constant', constant_values=np.nan)
            
        
            im = ax.scatter(currx, curry, c=tag[0:len(currx)], cmap=cmap, norm=norm, vmax=nframes_in_trial, s=1, alpha=0.1)

        if col == 0:
            ax.set(ylabel=row_label)
        if row == nr-1:
            ax.set(xlabel = col_label)
        #ax.set_xlim(minval, maxval)
        #ax.set_ylim(minval, maxval)
        ax.set(aspect='equal') #adjustable='box-forced', aspect='equal')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))



    















#%%



#######
y_test = test_dataset['ylabels']
if test_data_type == 'meanstim':
    y_test = np.reshape(y_test, (test_dataset['run_info'][()]['ntrials_total'], test_dataset['run_info'][()]['nframes_per_trial']))[:,0]

sconfigs_test = test_dataset['sconfigs'][()]
runinfo_test = test_dataset['run_info'][()]
trial_nframes_test = runinfo_test['nframes_per_trial']

all_tsecs = np.reshape(test_dataset['tsecs'][:], (sum(runinfo_test['ntrials_by_cond'].values()), trial_nframes_test))
print all_tsecs.shape

stimtype = 'gratings'


data_subset = 'all' #'single' # 'xpos-5' # 'alltrials' #'xpos-5'
const_trans = '' #'xpos'
trans_value = '' #-5

class_name = decoder

if 'all' in data_subset:
    if stimtype == 'gratings':
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])  # y_test.copy()
        colorvals = sns.color_palette("cubehelix", len(svc.classes_))

    else:
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])
        colorvals = sns.color_palette("PRGn", len(svc.classes_))

    test_data = X_test.copy()
    tsecs = all_tsecs.copy()
# Get subset of test data (position matched, if trained on y-rotation at single position):
elif 'single' in data_subset:
    assert len([i for i in sconfigs_test.keys() if sconfigs_test[i][const_trans]==trans_value]) > 0, "Specified stim config does not exist (const_trans=%s, trans_value=%s)" % (str(const_trans), str(trans_value))
    #included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv][const_trans]==trans_value]) # and sconfigs_test[yv]['yrot']==1])
    included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv][const_trans]==trans_value \
                         and sconfigs_test[yv]['morphlevel'] > 0]) # and sconfigs_test[yv]['yrot']==1])

    test_labels = np.array([sconfigs_test[c][class_name] for c in y_test[included]])
    test_data = X_test[included, :]
    sub_tsecs = test_dataset['tsecs'][included]
    tsecs = np.reshape(sub_tsecs, (len(sub_tsecs)/trial_nframes_test, trial_nframes_test))
    #tsecs = all_tsecs[included,:] # Only need 1 column, since all ROIs have the same tpoints
print "TEST: %s, labels: %s" % (str(test_data.shape), str(test_labels.shape))


#object_ids = [label for label in list(set(test_labels)) if label in train_labels]
object_ids = sorted([label for label in list(set(test_labels))])

#object_ids = [-1, 0, 53, 22]

print "LABELS:", object_ids


###############################################################################

mean_pred = {}
sem_pred = {}
predicted = []
for tru in sorted(list(set(test_labels))):
    trial_ixs = np.where(test_labels==tru)[0]
    curr_test = X_test[trial_ixs,:]
    y_proba = clf.predict_proba(curr_test)
    
    means_by_class = np.mean(y_proba, axis=0)
    stds_by_class = np.std(y_proba, axis=0)
    
    
    mean_pred[tru] = means_by_class
    sem_pred[tru] = stds_by_class

fig, axes = pl.subplots(1, len(mean_pred.keys()))
for ax,morph in zip(axes.flat, sorted(mean_pred.keys())):
    ax.errorbar(xrange(len(mean_pred[morph])), mean_pred[morph], yerr=sem_pred[morph])
    
    
    
# Get probabilities as a function of time #####################################
colorvals = sns.color_palette("PRGn", len(clf.classes_))
if len(colorvals) == 3:
    # Replace the middle!
    colorvals[1] = 'gray'

mean_pred = {}
sem_pred = {}
all_pred= {}
timesecs= {}

for curr_obj_id in object_ids:
    
    frame_ixs = np.where(test_labels==curr_obj_id)[0]
    n_test_trials = int(len(frame_ixs) / float(trial_nframes_test))
    #print curr_obj_id, n_test_trials
    
    test_trialmat_fixs = np.reshape(frame_ixs, (n_test_trials, trial_nframes_test))  # Reshape indices into nframes x ntrials array:
    #data_r = np.reshape(test_data[frame_ixs,:], (n_test_trials, trial_nframes_test, test_data.shape[-1]))
    predicted = []
    for ti in range(n_test_trials):
        true_labels = test_labels[test_trialmat_fixs[ti]]  # True labels for frames of current trial
        test = test_data[test_trialmat_fixs[ti], :]        # Test data (predict labels for each frame of current trial) -- nframes_per_trial x nrois
        tpoints = tsecs[ti, :] #tsecs[test_trialmat_fixs[ti]]            # Time stamps for frames of current trial
        y_proba = clf.predict_proba(test)                  # Predict label
        #pl.plot(tpoints, y_proba[:,0], color=colorvals[0])
        #pl.plot(tpoints, y_proba[:,4], color=colorvals[4])
        predicted.append(y_proba)
    predicted = np.array(predicted)
    mean_pred[curr_obj_id] = np.mean(predicted, axis=0)
    sem_pred[curr_obj_id] = stats.sem(predicted, axis=0)
    timesecs[curr_obj_id] = tpoints #np.mean(tsecs, axis=1) #np.array(tpoints)
    #all_pred[curr_obj_id] = np.array(predicted)
    
    
# PLOT ########################################################################


    
sns.set_style('white')

# Plot AVERAGE of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(20,8), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        tpoints = timesecs[obj]
        stim_on = np.where(tpoints==0)[0]
        stim_bar = np.array([tpoints[int(stim_on)], tpoints[int(stim_on+round(runinfo_test['nframes_on']))]])
        
        preds_mean = mean_pred[obj][:, ci]
        preds_sem = sem_pred[obj][:, ci]
        
        axes[pi].plot(tpoints, preds_mean, color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
        axes[pi].fill_between(tpoints, preds_mean-preds_sem, preds_mean+preds_sem, color=colorvals[ci], alpha=0.2)
        axes[pi].get_xaxis().set_visible(False)
        
        # stimulus bar:
        axes[pi].plot(stim_bar, np.ones(stim_bar.shape) * 0.3, 'k')
        
    axes[pi].set(title=obj)
    # Shrink current axis's height by 10% on the bottom
    box = axes[pi].get_position()
    axes[pi].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])
    if pi == 0:
        axes[pi].set(ylabel='avg p')
        

sns.despine(offset=True, bottom=True)

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1., -.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
pl.suptitle('Prob of trained classes on trial frames')

#output_dir = os.path.join(acquisition_dir, 'tests', 'figures')
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#pl.ylim(0.3, 0.7)

figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_%s_darkercols.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset, test_data_type)
print figname


classifier_dir = os.path.join(train_basedir, 'classifiers', classif_identifier)
pl.savefig(os.path.join(classifier_dir, figname))








#%%

roi_selector = 'all' #'all' #'selectiveanova' #'selective'
data_type = 'stat' #'zscore' #zscore' # 'xcondsub'
inputdata = 'meanstim'

testX, testy, test_labels = format_classifier_data(test_dataset, data_type=test_dtype, 
                                                       class_name=class_name,
                                                       subset=subset,
                                                       aggregate_type=aggregate_type,
                                                       const_trans=const_trans,
                                                       trnas_value=trans_value,
                                                       relabel=relabel)

print "Possible test labels:", test_labels
print "Known trained labels:", class_labels 

svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
cX_train, cX_test, cy_train, cy_test = train_test_split(cX_std, cy, test_size=0.25, shuffle=True)
svc.fit(cX_train, cy_train)


m100 = 22
choices = []
for tlabel in test_labels:
    if tlabel in class_labels:
        curr_ypred = svc.predict(cX_test)
        avg_pred = np.mean([1 if pred==tru else 0 for i,(pred,tru) in enumerate(zip(curr_ypred, cy_test))])
        
    else:
        tdata_ixs = np.where(testy == tlabel)[0]
        curr_ytest = testy[tdata_ixs]
        curr_xtest = testX[tdata_ixs,:]
        # Look at accuracy for each trial?
        curr_ypred = svc.predict(curr_xtest)
        avg_pred = np.mean([1 if p==m100 else 0 for p in curr_ypred])
    choices.append((tlabel, avg_pred))
    
# plot choice percentage:
pl.figure()
m100 = 22
if m100==22:
    choices[0] = (0, 1-choices[-1][1])


choices = morph_test['choices']
fig, ax = pl.subplots(1) #pl.figure();
pl.plot([c[0] for c in choices], [c[1] for c in choices], 'ro', markersize=10)
pl.ylabel('perc. choose %i' % m100)
ax.set_xticks([c[0] for c in choices])
pl.xlabel('morph level')
pl.ylim([0, 1])
sns.despine(offset=4, trim='bottom')

pl.savefig(os.path.join(classifier_dir, 'pchoose_%i_test25perc.pdf' % m100))

from scipy.optimize import curve_fit
def sigmoid(x, x0, k):
     #y = chance + (1 - chance) / (1 + np.exp(-k*(x-x0)))
     y = 0.5 + (0.5)/(1. + np.exp(-k*(x-x0)))
     return y

xs = [c[0] for c in choices]
tofit = [c[1] for c in choices]

popt, pcov = curve_fit(sigmoid, xs, tofit)
print popt
x = np.linspace(-1, 15, 50)
chance_level = 0.5
y = sigmoid(x, *popt)

pl.figure()
pl.plot(xs, tofit, 'ro', markersize=10, label='data')
pl.plot(x,y, 'k', linewidth=2, label='fit')
pl.ylim(0, 1.05)
pl.legend(loc='best')
pl.show()


morph_test = {'train_x': cX_train,
              'train_y': cy_train,
              'test_x': cX_test,
              'test_y': cy_test,
              'choices': choices,
              'svc': svc}

with open(os.path.join(classifier_dir, 'pchoose_%i_samedataset.pkl'), 'wb') as f:
    pkl.dump(morph_test, f, protocol=pkl.HIGHEST_PROTOCOL)

with open(os.path.join(classifier_dir, 'pchoose_%i_samedataset.pkl' % m100), 'rb') as f:
    morph_test = pkl.load(f)
#%




#%%
# Try using XPOS data for test set:
opts_static = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180523', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_run1', '-t', 'traces001',
           '-n', '1']

opts_dynamic = ['-D', '/mnt/odyssey', '-i', 'CE077', '-S', '20180602', '-A', 'FOV1_zoom1x',
           '-T', 'np_subtracted', '--no-pupil',
           '-R', 'blobs_dynamic_run6', '-t', 'traces001',
           '-n', '1']

traceid_dir_static = util.get_traceid_dir(opts_static)
data_basedir_static = os.path.join(traceid_dir_static, 'data_arrays')


svc = LinearSVC(random_state=0, dual=dual, multi_class='ovr', C=big_C)
svc.fit(cX_std, cy)


# First check if processed datafile exists:
static_fpath = os.path.join(data_basedir_static, 'datasets.npz')
static = np.load(static_fpath)


testdata_X = static['meanstim']
testdata_Xstd = StandardScaler().fit_transform(testdata_X)
testdata_y = np.reshape(static['ylabels'], (static['run_info'][()]['ntrials_total'], static['run_info'][()]['nframes_per_trial']))[:,0]
sconfigs_test = static['sconfigs'][()]

test_morphs = sorted(list(set(np.array([sconfigs_test[cv]['morphlevel'] for cv in sconfigs_test.keys()]))))


pos_values = [-15, -10, -5, 0, 5]
all_choices = {}
choices = dict((k, []) for k in test_morphs)

for pos in pos_values:
    testX, testy = group_classifier_data(testdata_Xstd, testdata_y, 'morphlevel', sconfigs_test, 
                                   subset=None,
                                   aggregate_type='single', 
                                   const_trans='xpos', 
                                   trans_value=pos, 
                                   relabel=False)

    testy = np.array([sconfigs_test[cv]['morphlevel'] for cv in testy])
    
    test_labels = sorted(list(set(testy)))
    print "Possible test labels:", test_labels
    print "Known trained labels:", class_labels 

    for tlabel in test_labels:
        tdata_ixs = np.where(testy == tlabel)[0]
        curr_ytest = testy[tdata_ixs]
        curr_xtest = testX[tdata_ixs,:]
        # Look at accuracy for each trial?
        curr_ypred = svc.predict(curr_xtest)
        choices[tlabel].append(curr_ypred)

averages = dict((k, {}) for k in choices.keys())
for mlevel in choices.keys():
    mean_choices = [np.mean([1 if p==m100 else 0 for p in choices[mlevel][posi]]) for posi in range(len(choices[mlevel]))]
    averages[mlevel]['mean'] = np.mean(mean_choices)
    averages[mlevel]['sem'] = stats.sem(mean_choices)

# plot choice percentage:
means = [averages[mlevel]['mean'] for mlevel in sorted(averages.keys())]
sems = [averages[mlevel]['sem'] for mlevel in sorted(averages.keys())]


fig, ax = pl.subplots(1) #pl.figure();

pl.errorbar(sorted(averages.keys()), means, yerr=sems)
pl.ylabel('perc. choose %i' % m100)
ax.set_xticks(sorted(averages.keys()))
pl.xlabel('morph level')
pl.ylim([0.25, 0.75])
sns.despine(offset=4, trim=True)
pl.savefig(os.path.join(classifier_dir, 'pchoose_%i_xposdata_avgxpos.pdf' % m100))

#%%

#%% Assign data:


X_test = test_dataset[test_data_type]

#X_test = smoothed_X
X_test = np.array(smoothed_X)
X_test = X_test[:, xrange(0, cX_std.shape[-1])]


X_test = StandardScaler().fit_transform(X_test)
y_test = test_dataset['ylabels']
if test_data_type == 'meanstim':
    y_test = np.reshape(y_test, (test_dataset['run_info'][()]['ntrials_total'], test_dataset['run_info'][()]['nframes_per_trial']))[:,0]

sconfigs_test = test_dataset['sconfigs'][()]
runinfo_test = test_dataset['run_info'][()]
trial_nframes_test = runinfo_test['nframes_per_trial']

all_tsecs = np.reshape(test_dataset['tsecs'][:], (sum(runinfo_test['ntrials_by_cond'].values()), trial_nframes_test))
print all_tsecs.shape




data_subset = 'all' # 'xpos-5' # 'alltrials' #'xpos-5'

if 'all' in data_subset:
    if stimtype == 'gratings':
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])  # y_test.copy()
        colorvals = sns.color_palette("cubehelix", len(svc.classes_))

    else:
        test_labels = np.array([sconfigs_test[c][class_name] for c in y_test])
        colorvals = sns.color_palette("PRGn", len(svc.classes_))

    test_data = X_test.copy()
    tsecs = all_tsecs.copy()
# Get subset of test data (position matched, if trained on y-rotation at single position):
else:
    included = np.array([yi for yi, yv in enumerate(y_test) if sconfigs_test[yv]['xpos']==-5]) # and sconfigs_test[yv]['yrot']==1])
    test_labels = np.array([sconfigs_test[c][class_name] for c in y_test[included]])
    test_data = X_test[included, :]
    sub_tsecs = test_dataset['tsecs'][included]
    tsecs = np.reshape(sub_tsecs, (len(sub_tsecs)/trial_nframes_test, trial_nframes_test))
    #tsecs = all_tsecs[included,:] # Only need 1 column, since all ROIs have the same tpoints
print "TEST: %s, labels: %s" % (str(test_data.shape), str(test_labels.shape))


#object_ids = [label for label in list(set(test_labels)) if label in train_labels]
object_ids = sorted([label for label in list(set(test_labels))])

#object_ids = [-1, 0, 53, 22]

print "LABELS:", object_ids

mean_pred = {}
sem_pred = {}
predicted = []
for tru in sorted(list(set(test_labels))):
    trial_ixs = np.where(test_labels==tru)[0]
    curr_test = X_test[trial_ixs,:]
    y_proba = clf.predict_proba(curr_test)
    
    means_by_class = np.mean(y_proba, axis=0)
    stds_by_class = np.std(y_proba, axis=0)
    
    
    mean_pred[tru] = means_by_class
    sem_pred[tru] = stds_by_class

fig, axes = pl.subplots(1, len(mean_pred.keys()))
for ax,morph in zip(axes.flat, sorted(mean_pred.keys())):
    ax.errorbar(xrange(3), mean_pred[morph], yerr=sem_pred[morph])
    
    
    
# Get probabilities as a function of time #####################################
colorvals = sns.color_palette("PRGn", len(clf.classes_))
if len(colorvals) == 3:
    # Replace the middle!
    colorvals[1] = 'gray'

mean_pred = {}
sem_pred = {}
all_pred= {}
timesecs= {}

for curr_obj_id in object_ids:
    
    frame_ixs = np.where(test_labels==curr_obj_id)[0]
    n_test_trials = int(len(frame_ixs) / float(trial_nframes_test))
    #print curr_obj_id, n_test_trials
    
    test_trialmat_fixs = np.reshape(frame_ixs, (n_test_trials, trial_nframes_test))  # Reshape indices into nframes x ntrials array:
    #data_r = np.reshape(test_data[frame_ixs,:], (n_test_trials, trial_nframes_test, test_data.shape[-1]))
    predicted = []
    for ti in range(n_test_trials):
        true_labels = test_labels[test_trialmat_fixs[ti]]  # True labels for frames of current trial
        test = test_data[test_trialmat_fixs[ti], :]        # Test data (predict labels for each frame of current trial) -- nframes_per_trial x nrois
        tpoints = tsecs[ti, :] #tsecs[test_trialmat_fixs[ti]]            # Time stamps for frames of current trial
        y_proba = clf.predict_proba(test)                  # Predict label
        #pl.plot(tpoints, y_proba[:,0], color=colorvals[0])
        #pl.plot(tpoints, y_proba[:,4], color=colorvals[4])
        predicted.append(y_proba)
    predicted = np.array(predicted)
    mean_pred[curr_obj_id] = np.mean(predicted, axis=0)
    sem_pred[curr_obj_id] = stats.sem(predicted, axis=0)
    timesecs[curr_obj_id] = tpoints #np.mean(tsecs, axis=1) #np.array(tpoints)
    #all_pred[curr_obj_id] = np.array(predicted)
    
    
# PLOT ########################################################################
    
sns.set_style('white')
# Plot AVERAGE of movie conditions:
fig, axes = pl.subplots(1, len(object_ids), figsize=(20,8), sharey=True)
for pi,obj in enumerate(object_ids):
    for ci in range(mean_pred[obj].shape[1]):
        tpoints = timesecs[obj]
        preds_mean = mean_pred[obj][:, ci]
        preds_sem = sem_pred[obj][:, ci]
        
        axes[pi].plot(tpoints, preds_mean, color=colorvals[ci], label=clf.classes_[ci], linewidth=2)
        axes[pi].fill_between(tpoints, preds_mean-preds_sem, preds_mean+preds_sem, color=colorvals[ci], alpha=0.2)
    axes[pi].set(title=obj)
    # Shrink current axis's height by 10% on the bottom
    box = axes[pi].get_position()
    axes[pi].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.8])
    if pi == 0:
        axes[pi].set(ylabel='avg p')
sns.despine(offset=True, trim=True)

# Put a legend below current axis
pl.legend(loc='lower center', bbox_to_anchor=(-1., -.3),
          fancybox=True, shadow=False, ncol=len(clf.classes_))
pl.suptitle('Prob of trained classes on trial frames')

#output_dir = os.path.join(acquisition_dir, 'tests', 'figures')
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

pl.ylim(0, 0.7)

figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_%s_darkercols.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset, test_data_type)
print figname

pl.savefig(os.path.join(classifier_dir, figname))

#%%






pl.figure()
dfunc = svc.decision_function(cX_std)
sns.heatmap(dfunc, cmap='hot')
pl.title('decision function')
figname = 'TRAIN_%s_%s_C_%s_decisionfunction.png' % (train_runid, train_traceid, classif_identifier)
pl.savefig(os.path.join(output_dir, figname))



pl.figure()
sns.heatmap(svc.decision_function(test_data), cmap='hot')
pl.title('decision function -- test data')
figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_decisionfunction.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset)
pl.savefig(os.path.join(output_dir, figname))





pl.figure()
sns.heatmap(svc.decision_function(stimvalues.T), cmap='hot')
pl.title('decision function - mean stim values (test set)')
figname = 'TRAIN_%s_%s_C_%s_TEST_%s_%s_%s_decisionfunction_meanstim_testvals.png' % (train_runid, train_traceid, classif_identifier, test_runid, test_traceid, data_subset)
pl.savefig(os.path.join(output_dir, figname))


