#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:03:10 2018

@author: julianarhee
"""

stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
stim_frames = np.arange(stim_on, stim_on+nframes_on)
nframes_in_trial = np.squeeze(all_preds[config]).shape[0]


pres = mean_pred[config][0:40, :].flatten()
stims = mean_pred[config][stim_frames[100:140], :].flatten()
posts = mean_pred[config][-40:, :].flatten()

fig, axes = pl.subplots(1,3, sharex=True, sharey=True)

sns.distplot(pres, ax=axes[0])
sns.distplot(stims, ax=axes[1])
sns.distplot(posts, ax=axes[2])

fig, ax = pl.subplots(1)
ax.plot(np.arange(0, stim_frames[0]), mean_pred[config][0:stim_frames[0], :],
                    color='k', linewidth=mean_lw, alpha=0.5)

ax.plot(stim_frames, mean_pred[config][stim_frames, :], 
                    color='k', linewidth=mean_lw, alpha=1.0)

ax.plot(np.arange(stim_frames[-1], nframes_in_trial), mean_pred[config][stim_frames[-1]:, :], 
                    color='k', linewidth=mean_lw, alpha=0.5)

config_ix = 5 #int(config[-3:])-1
ax.plot(range(nframes_in_trial), mean_pred[config][:, config_ix], color='r')



#%%
decode_ix = 5
pres = all_preds[config][0:stim_frames[0], decode_ix].flatten()
stims = all_preds[config][stim_frames, decode_ix].flatten()
posts = all_preds[config][stim_frames[-1]+45:, decode_ix].flatten()

sns.distplot(pres, label='pre'); sns.distplot(stims, label='stim'); sns.distplot(posts, label='post')
pl.legend()

sampling = np.copy(pres)

def shuff2D(sampling):
    d1, d2 = sampling.shape
    random.shuffle(sampling.ravel())
    sampling = sampling.reshape(d1, d2)
    return sampling

def shuff(sampling):
    random.shuffle(sampling)
    return sampling

sns.distplot(sampling, label='true')
for ndraws in range(niters):
    sampling = np.copy(pres)
    rand_pres = [np.max([shuff(sampling)[0] for i in range(ndraws)]) for n in range(niters)]
    sns.distplot(rand_pres, label=str(ndraws))
pl.legend()



# Random distN of ndraws (rand w/ replacement), get max.
# Calculate the probability of true max:
niters = 1000
sampling = all_preds[config][0:stim_frames[0]:, decode_ix].copy().flatten()

#fig, axes = pl.subplots(4, 5)
#for ax,trial in zip(axes.flat, range(sampling.shape[-1])):
#    plot_acf(sampling[:, trial], ax=ax)
    
#ndraws = all_preds[config][stim_frames[-1]:, decode_ix].shape[0]

# Tile original array, so it is ndraws X nframes (each row is a replicate)
# Randomly shuffle array, draw a value - repeat until generate "fake" trace array
# Find max value, then repeat for niters.

nframes_sim = len(mean_pred[config][stim_frames[-1]:, decode_ix])
bootmax = np.array([np.array(shuff(sampling)[0:nframes_sim]).max() for _ in range(niters)])



fig, ax = pl.subplots(1)
sns.distplot(bootmax, ax=ax, hist=True, kde=False, bins=100)


# Plot decoding traces, plus p-value bars:
trial_lw = 0.2
mean_lw = 2
fig, ax = pl.subplots(1, figsize=(16,4))

ax.plot(np.arange(0, stim_frames[0]), all_preds[config][0:stim_frames[0], decode_ix],
                    color='k', linewidth=trial_lw, alpha=1)
ax.plot(stim_frames, all_preds[config][stim_frames, decode_ix], 
                    color='k', linewidth=trial_lw, alpha=1)
ax.plot(np.arange(stim_frames[-1], nframes_in_trial), all_preds[config][stim_frames[-1]:, decode_ix], 
                    color='k', linewidth=trial_lw, alpha=1)

ax.plot(np.arange(0, stim_frames[0]), mean_pred[config][0:stim_frames[0], decode_ix],
                    color='k', linewidth=mean_lw, alpha=0.5)
ax.plot(stim_frames, mean_pred[config][stim_frames, decode_ix], 
                    color='k', linewidth=mean_lw, alpha=1.0)
ax.plot(np.arange(stim_frames[-1], nframes_in_trial), mean_pred[config][stim_frames[-1]:, decode_ix], 
                    color='k', linewidth=mean_lw, alpha=0.5)

# For each value of the mean decoding trace, plot bar colored by p-value:
ylim = 0.5
ax.set_ylim([0, ylim])

pvals = [(v, len(bootmax[bootmax>=v])/float(niters)) for v in mean_pred[config][stim_frames[-1]:, decode_ix]]
for fi,p in enumerate(pvals):
    if p[1] < 0.05:
        ax.axvline(stim_frames[-1]+fi, linewidth=3, color='magenta', ymin=0, ymax=p[0]/ylim, alpha=0.3)
    else:
        ax.axvline(stim_frames[-1]+fi, linewidth=3, color='k', ymin=0, ymax=p[0]/ylim, alpha=0)

#%%
# Plot each CLASS's probability on a subplot:
# -----------------------------------------------------------------------------
niters = 10000

colorvals = sns.color_palette("hls", len(svc.classes_))
stim_on = list(set(labels_df[labels_df['config']==config]['stim_on_frame']))[0]
nframes_on = list(set(labels_df[labels_df['config']==config]['nframes_on']))[0]
stim_frames = np.arange(stim_on, stim_on+nframes_on)

angle_step = list(set(np.diff(train_labels)))[0]
if test_configs[config]['direction'] == 1:  # CW, values should be decreasing
    class_list = sorted(train_labels, reverse=True)
    shift_sign = -1
else:
    class_list = sorted(train_labels, reverse=False)
    shift_sign = 1
    
start_angle_ix = class_list.index(test_configs[config]['ori'])
class_list = np.roll(class_list, shift_sign*start_angle_ix)

class_indices = [[v for v in svc.classes_].index(c) for c in class_list]

if test_configs[config]['stim_dur'] == quarter_dur:
    nclasses_shown = int(len(class_indices) * (1/4.)) + 1
elif test_configs[config]['stim_dur'] == half_dur:
    nclasses_shown = int(len(class_indices) * (1/2.)) + 1
else:
    nclasses_shown = len(class_indices)
ordered_indices = np.array(class_indices[0:nclasses_shown])
ordered_colors = [colorvals[c] for c in ordered_indices]

# Create psuedo-continue cmap:
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, basestring): low = c(low)
    if isinstance(high, basestring): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])

cgradient = diverge_map(low = ordered_colors[0], high = ordered_colors[-1])
           
cgradient = make_colormap(ordered_colors)
 
            
fig, axes = pl.subplots(len(class_list), 1, figsize=(6,15))
        
for lix, (class_label, class_index) in enumerate(zip(class_list, class_indices)):
    nframes_sim = len(mean_pred[config][stim_frames[-1]:, class_index])
    sampling = mean_pred[config][0:stim_frames[0]:, class_index].copy().flatten()
    bootmax = np.array([np.array(shuff(sampling)[0:nframes_sim]).max() for _ in range(niters)])


#    axes[lix].plot(np.arange(0, stim_frames[0]), all_preds[config][0:stim_frames[0], class_index],
#                        color='k', linewidth=trial_lw, alpha=1)
#    axes[lix].plot(stim_frames, all_preds[config][stim_frames, class_index], 
#                        color=colorvals[class_index], linewidth=trial_lw, alpha=1)
#    axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), all_preds[config][stim_frames[-1]:, class_index], 
#                        color='k', linewidth=trial_lw, alpha=1)
#    
    axes[lix].plot(np.arange(0, stim_frames[0]), mean_pred[config][0:stim_frames[0], class_index],
                        color='k', linewidth=mean_lw, alpha=0.5)
    axes[lix].plot(stim_frames, mean_pred[config][stim_frames, class_index], 
                        color=colorvals[class_index], linewidth=mean_lw, alpha=1.0)
    axes[lix].plot(np.arange(stim_frames[-1], nframes_in_trial), mean_pred[config][stim_frames[-1]:, class_index], 
                        color='k', linewidth=mean_lw, alpha=0.5)
    
    # For each value of the mean decoding trace, plot bar colored by p-value:
    ylim = 0.6
    axes[lix].set_ylim([0, ylim])
    
    pvals = [(v, len(bootmax[bootmax>=v])/float(niters)) for v in mean_pred[config][stim_frames[-1]:, class_index]]
    for fi,p in enumerate(pvals):
        if p[1] < 0.05:
            axes[lix].axvline(stim_frames[-1]+fi, linewidth=3, color='k', ymin=0, ymax=p[0]/ylim, alpha=0.2)

        # Plot chance line:
        chance = 1/len(class_list)
        axes[lix].plot(np.arange(0, nframes_in_trial), np.ones((nframes_in_trial,))*chance, 'k--', linewidth=0.5)
        

sns.despine(trim=True, offset=4)

for lix in range(len(class_list)):
    # Create color bar:
    cy = np.ones(stim_frames.shape) * axes[lix].get_ylim()[0]/2.0
    z = stim_frames.copy()
    points = np.array([stim_frames, cy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #cmap = ListedColormap(ordered_colors)
    lc = LineCollection(segments, cmap=cgradient)
    lc.set_array(z)
    lc.set_linewidth(8)
    axes[lix].add_collection(lc)
    
    if lix == len(class_list)-1:
        axes[lix].set_xticks((stim_on, stim_on + framerate))
        axes[lix].set_xticklabels([0, 1])
        axes[lix].set_xlabel('sec', horizontalalignment='right', x=0.25)        
        for axside in ['top', 'right']:
            axes[lix].spines[axside].set_visible(False)
        sns.despine(trim=True, offset=4, ax=axes[lix])

    else:
        axes[lix].axes.xaxis.set_ticks([])
        for axside in ['bottom', 'top', 'right']:
            axes[lix].spines[axside].set_visible(False)
        axes[lix].axes.xaxis.set_visible(False) #([])
        sns.despine(trim=True, offset=4, ax=axes[lix])
        
    axes[lix].set_ylabel('prob (%i)' % class_list[lix])