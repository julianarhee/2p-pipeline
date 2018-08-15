#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:15:16 2018

@author: juliana
"""



def wrapped_gaussian(theta, *params):
    (Rpreferred, Rnulll, theta_preferred, sigma, offset) = params
#    a = theta - theta_preferred 
#    b = theta + 180 - theta_preferred
#    wrap_a = min( [a, a-360, a+360] )
#    wrap_b = min( [b, b-360, b+360] )

    wrap_as = theta[:, 0]
    wrap_bs = theta[:, 1]
    pref_term = Rpreferred * np.exp( - (wrap_as**2) / (2.0 * sigma**2.0) )
    null_term = Rnull * np.exp( - (wrap_bs**2) / (2.0 * sigma**2.0) )
    R = offset + pref_term +  null_term
    
    return R


def wrapped_curve(theta, *params):
    (Rpreferred, theta_preferred, sigma) = params
    v = 0
    for n in [-2, 2]:
        v += np.exp( -((theta - theta_preferred + 180*n)**2) / (2 * sigma**2) )
    return Rpreferred * v


def von_mises_double(theta, *params):
    (Rpreferred, Rnull, theta_preferred, k1, k2) = params
    pterm = Rpreferred * np.exp(k1 * ( np.cos( thetas*(math.pi/180.)- theta_preferred*(math.pi/180.) ) - 1))
    nterm = Rnull * np.exp(k2 * ( np.cos( theta*(math.pi/180.)- theta_preferred*(math.pi/180.) ) - 1))
    return pterm + nterm


def von_mises(theta, *params):
    (Rpreferred, theta_preferred, sigma, offset) = params
    R = Rpreferred * np.exp( sigma * (np.cos( 2*(theta*(math.pi/180.) - theta_preferred*(math.pi/180.)) ) - 1) )
    return R + offset


#def double_gaussian( x, params ):
#    (c1, mu1, sigma1, c2, mu2, sigma2) = params
#    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
#          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
#    return res

def double_gaussian_fit( params ):
    fit = wrapped_gaussian( xvals, params )
    return (fit - y_proc)


from scipy.optimize import curve_fit
from scipy.optimize import leastsq


#%%

# TRY von mises for NON-drifiting (0, 180)

upsampled_thetas = np.linspace(thetas[0], thetas[-1], num=len(thetas)*100)

drifting = False
thetas = [train_configs[cf]['ori'] for cf in config_list]


nrows = 5
ncols = 10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)

nrois = responses.shape[-1]
#ridx = 2

for ridx, ax in zip(range(nrois), axes.flat):
    tuning = responses[:, ridx]
    offset = np.mean(offsets[:, ridx]) #np.mean(offsets[:, ridx])
    max_ix = np.where(tuning==tuning.max())[0][0]
    Rpreferred = tuning[max_ix]
    theta_preferred = thetas[max_ix]
    
    if drifting:
        if theta_preferred >=180:
            theta_null = theta_preferred - 180
        else:
            theta_null = theta_preferred + 180
        null_ix = thetas.index(theta_null)
        Rnull = tuning[null_ix]
    
        print "Rpref: %.3f (%i) | Rnull: %.3f (%i)" % (Rpreferred, theta_preferred, Rnull, theta_null)
    
    else:
        print "Rpref: %.3f (%i) " % (Rpreferred, theta_preferred)
    
    #pl.figure()
    sigma_diffs = []
    test_sigmas = np.arange(5, 60, 5)
    for sigma in np.arange(5, 30, 5):
        
        try:
            popt=None; pcov=None;
            params = [Rpreferred, theta_preferred, sigma, offset]
            popt, pcov = curve_fit(von_mises, thetas, tuning, p0=params )
            sigma_diffs.append(np.abs(popt[2] - sigma))
            
            y_fit = von_mises(upsampled_thetas, *popt)
        except Exception as e:
            print "%i no fit" % ridx
        #pl.plot(thetas, tuning, 'ko')
        #pl.plot(upsampled_thetas, y_fit, label=sigma)
        
    #pl.legend()
    if popt is not None:
        best_ix = sigma_diffs.index(min(sigma_diffs))
        best_sigma = test_sigmas[best_ix]  
        print best_sigma
    
        params = [Rpreferred, theta_preferred, best_sigma, offset]
        popt, pcov = curve_fit(von_mises, thetas, tuning, p0=params )
        sigma_diffs.append(np.abs(popt[2] - sigma))
        
        y_fit = von_mises(upsampled_thetas, *popt)
        ax.plot(thetas, tuning, 'ko')
        ax.plot(upsampled_thetas, y_fit, label=sigma)
        

#def get_wrap(thetas):
#    xvals = np.empty((len(thetas), 2))
#    #print xvals.shape
#    for ti, theta in enumerate(thetas):
#        a = theta - theta_preferred 
#        b = theta + 180 - theta_preferred
#        wrap_a = min( [a, a-360, a+360] )
#        wrap_b = min( [b, b-360, b+360] )
#        xvals[ti, :] = np.array([wrap_a, wrap_b])
#    #print xvals[:, 0]
#    #print xvals[:, 1]
#    return xvals
#   
#ofset = 0
#for sigma in np.arange(45*(math.pi/180)/4, 45*(math.pi/180)*2, 10*(math.pi/180)):
#    
#    params = [Rpreferred, Rnull, theta_preferred, sigma, offset]
#    popt, pcov = curve_fit(wrapped_gaussian, xvals, tuning, p0=params )
#    
#    y_fit = wrapped_gaussian(get_wrap(upsampled_thetas), *popt)
#    pl.figure()
#    pl.plot(thetas, tuning, 'ko')
#    pl.plot(upsampled_thetas, y_fit)


#%%

# Try bimodal with drifting:
    
upsampled_thetas = np.linspace(thetas[0], thetas[-1], num=100)

nrows = 5
ncols = 10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)

for ridx, ax in zip(range(nrois), axes.flat):
    
    tuning = responses[:, ridx]
    max_ix = np.where(tuning==tuning.max())[0][0]
    Rpreferred = tuning[max_ix]
    theta_preferred = thetas[max_ix]
#    if theta_preferred >=180:
#        theta_null = theta_preferred - 180
#    else:
#        theta_null = theta_preferred + 180
#    null_ix = thetas.index(theta_null)
#    Rnull = tuning[null_ix]
    print "%i - Rpref: %.3f (%i)" % (ridx, Rpreferred, theta_preferred)
    
    
    # Find best sigma:
    k1_fits=[]; k2_fits=[];
    #test_sigmas = np.arange(2.5*(math.pi/180), 45*(math.pi/180)*2, 2.5*(math.pi/180))
    test_sigmas = np.arange(5, 45*2, 5)
    for k in test_sigmas:
        try:
            popt=None; pcov=None;
            k1 = k; k2 = k;
            params = [Rpreferred, theta_preferred, k1, k2]
            popt, pcov = curve_fit(bimodal_curve, np.array(thetas), tuning, p0=params )
            k1_fits.append(k1-popt[3])
            k2_fits.append(k2-popt[4])        
        except Exception as e:
            print "%i - NO FIT" % ridx
            continue
    pl.legend()  
    
    if popt is not None:
        small_k = min([min(k1_fits), min(k2_fits)])
        if small_k in k1_fits:
            best_k = k1_fits.index(small_k)
        else:
            best_k = k2_fits.index(small_k)
        #print test_sigmas[best_k] * (180./math.pi)
        
        del popt
        del pcov
        best_sigma = test_sigmas[best_k] 
        k1 = best_sigma; k2 = best_sigma;
        params = [Rpreferred, Rnull, theta_preferred, k1, k2]
        popt, pcov = curve_fit(bimodal_curve, np.array(thetas), tuning, p0=params , maxfev=5000)
        y_fit = bimodal_curve(np.array(upsampled_thetas), *popt)
        ax.plot(thetas, tuning, 'ko')
        ax.plot(upsampled_thetas, y_fit) #, label=k*(180./math.pi))
        ax.set_title("sigma = %.2f" % best_sigma, fontsize=8)
        ax.set_xticks(thetas)

sns.despine(trim=True, offset=4)
    



#%%

# Thin plate spline interp?

#ridx = 15
#tuning = responses[:, ridx]
#from scipy.interpolate import splprep, splev
#
#
#tck, u = splprep([thetas, tuning], k=5, t=-1, s=1000)
#new_points = splev(u, tck)
#
#fig, ax = pl.subplots()
#ax.plot(thetas, tuning, 'ro')
#ax.plot(new_points[0], new_points[1], 'r-')
#pl.show()

# Just fit a polynomial...

def _polynomial(x, *p):
    """Polynomial fitting function of arbitrary degree."""
    poly = 0.
    for i, n in enumerate(p):
        poly += n * x**i
    return poly

#p0 = np.ones(6,)
#
#coeff, var_matrix = curve_fit(_polynomial, thetas, tuning, p0=p0)
#
#yfit = [_polynomial(xx, *tuple(coeff)) for xx in upsampled_thetas] # I'm sure there is a better
#                                                    # way of doing this
#pl.figure()
#pl.plot(thetas, tuning, 'ko', label='Test data', )
#pl.plot(upsampled_thetas, yfit, label='fitted data')
#



# TRY von mises for NON-drifiting (0, 180)
upsampled_thetas = np.linspace(thetas[0], thetas[-1], num=len(thetas)*100)

drifting = False
thetas = [train_configs[cf]['ori'] for cf in config_list]


nrows = 5
ncols = 10

nrois = responses.shape[-1]
fig, axes = pl.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)

nrois = responses.shape[-1]
#ridx = 2

interp_curves = []
for ridx, ax in zip(range(nrois), axes.flat):
    tuning = responses[:, ridx]
    offset = np.mean(offsets[:, ridx]) #np.mean(offsets[:, ridx])
    max_ix = np.where(tuning==tuning.max())[0][0]
    Rpreferred = tuning[max_ix]
    theta_preferred = thetas[max_ix]
    
    if drifting:
        if theta_preferred >=180:
            theta_null = theta_preferred - 180
        else:
            theta_null = theta_preferred + 180
        null_ix = thetas.index(theta_null)
        Rnull = tuning[null_ix]
    
        print "Rpref: %.3f (%i) | Rnull: %.3f (%i)" % (Rpreferred, theta_preferred, Rnull, theta_null)
    
    else:
        print "Rpref: %.3f (%i) " % (Rpreferred, theta_preferred)
            
    try:
        
        coeff=None; var_matrix=None; yfit=None;
        p0 = np.ones(5,)
        coeff, var_matrix = curve_fit(_polynomial, thetas, tuning, p0=p0)
        y_fit = [_polynomial(xx, *tuple(coeff)) for xx in upsampled_thetas] # I'm sure there is a better
                                                            # way of doing this
        interp_curves.append(y_fit)
    except Exception as e:
        print "%i no fit" % ridx
    if coeff is not None:
        ax.plot(thetas, tuning, 'ko')
        ax.plot(upsampled_thetas, y_fit, label=sigma)
    
    pl.savefig(os.path.join(sim_dir, '%s_polynomial_fits.png' % roiset))

interp_curves = np.vstack(interp_curves) # Nrois x N-sampled points
