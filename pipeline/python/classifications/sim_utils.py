import os
import glob
import pylab as pl

import numpy as np
import scipy.signal
import cv2
import matplotlib.patches as patches
import seaborn as sns

from scipy import ndimage

from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from pipeline.python import utils as putils


# Stimulus drawing functions

def draw_stimulus_to_screen(stimulus_im, size_deg=30., stim_pos=(0, 0),
                            pix_per_deg=16.05, resolution=[1080, 1920]):
    
    # Reshape stimulus to what it would be at size_deg
    im_r = resize_image_to_coords(stimulus_im, size_deg=size_deg)

    # Get extent of resized image, relative to stimulus coordinates
    stim_xpos, stim_ypos = stim_pos
    stim_extent=[-im_r.shape[1]/2. + stim_xpos, im_r.shape[1]/2. + stim_xpos, 
            -im_r.shape[0]/2. + stim_ypos, im_r.shape[0]/2. + stim_ypos]

    # Create array (dims=resolution)
    stim_screen = place_stimulus_on_screen(stimulus_im, stim_extent, 
                                             resolution=resolution)
    
    return stim_screen, stim_extent

def resize_image_to_coords(im, size_deg=30, pix_per_deg=16.05, aspect_scale=1.747):
    '''
    Take original image (in pixels) and scale it to specified size for screen.
    Return resized image in pixel space.
    '''
    print(pix_per_deg)
    ref_dim = max(im.shape)
    resize_factor = ((size_deg*pix_per_deg) / ref_dim ) / pix_per_deg
    scale_factor = resize_factor * aspect_scale
    
    imr = cv2.resize(im, None, fx=scale_factor, fy=scale_factor)
    
    return imr


def place_stimulus_on_screen(im, extent, resolution=[1080, 1920]):
    '''
    Place re-sized image (resize_image_to_coors()) onto the screen at specified res.
    extent: (xmin, xmax, ymin, ymax)
    ''' 
    lin_x, lin_y = putils.get_lin_coords(resolution=resolution)
    
    xx, yy = np.where(abs(lin_x-extent[0])==abs(lin_x-extent[0]).min())
    xmin=int(np.unique(yy))

    xx, yy = np.where(abs(lin_x-extent[1])==abs(lin_x-extent[1]).min())
    xmax=int(np.unique(yy))

    xx, yy = np.where(abs(lin_y-extent[2])==abs(lin_y-extent[2]).min())
    ymin = resolution[0] - int(np.unique(xx))

    xx, yy = np.where(abs(lin_y-extent[3])==abs(lin_y-extent[3]).min())
    ymax = resolution[0] - int(np.unique(xx))

    nw = xmax - xmin
    nh = ymax - ymin
    im_r2 = cv2.resize(im, (nw, nh))

    sim_screen = np.zeros(lin_x.shape)
    sim_screen[ymin:ymax, xmin:xmax] = np.flipud(im_r2)

    return sim_screen

def convert_fitparams_to_pixels(rid, curr_rfs, pix_per_deg=16.06,
                                resolution=[1080, 1920],
                                convert_params=['x0', 'y0', 'fwhm_x', 'fwhm_y', 'std_x', 'std_y']):
    '''
    RF fit params in degrees, convert to pixel space for drawing
    '''
    lin_x, lin_y = putils.get_lin_coords(resolution=res)

    # Get position
    ctx = curr_rfs['x0'][rid]
    cty = curr_rfs['y0'][rid]

    # Convert to deg
    _, yy = np.where(abs(lin_x-ctx)==abs(lin_x-ctx).min())
    x0=int(np.unique(yy))
    xx, _ = np.where(abs(lin_y-cty)==abs(lin_y-cty).min())
    y0=res[0]-int(np.unique(xx))

    # Get sigmax-y:
    sigx = curr_rfs['fwhm_x'][rid]
    sigy = curr_rfs['fwhm_y'][rid]

    sz_x = sigx*pix_per_deg #*.5
    sz_y = sigy*pix_per_deg #*.5
    
    theta = curr_rfs['theta'][rid]
    
    return x0, y0, sz_x, sz_y, theta

def load_stimuli(root='/n/home00/juliana.rhee', 
                 stimulus_path='Repositories/protocols/physiology/stimuli/images'):

    stimulus_dir = os.path.join(root, stimulus_path)
    # stimulus_dir = '/n/home00/juliana.rhee/Repositories/protocols/physiology/stimuli/images'

    # Get image paths:
    #object_list = ['D1', 'D2']
    object_list = ['D1', 'M14', 'M27', 'M53', 'M66', 'M9', 'M93', 'D2']
    
    image_paths = []
    for obj in object_list:
        stimulus_type = 'Blob_%s_Rot_y_fine' % obj
        image_paths.extend(glob.glob(os.path.join(stimulus_dir, stimulus_type, '*_y0.png')))
    print("%i images found for %i objects" % (len(image_paths), len(object_list)))
    assert len(image_paths)>0, "No stimuli in:\n  %s" % stimulus_dir
    images = {}
    for object_name, impath in zip(object_list, image_paths):
        im = cv2.imread(impath)
        images[object_name] = im[:, :, 0]
    print("im shape:", images['D1'].shape)
    
    return images



# Image processing

def get_bbox_around_nans(rpatch, replace_nans=True, return_indices=False):
    bb_xmax, bb_ymax = np.max(np.where(~np.isnan(rpatch)), 1)
    bb_xmin, bb_ymin = np.min(np.where(~np.isnan(rpatch)), 1)
    # print(bb_xmax, bb_ymax)

    tp = rpatch[bb_xmin:bb_xmax, bb_ymin:bb_ymax]
    bb_patch=tp.copy()
    if replace_nans:
        bb_patch[np.where(np.isnan(tp))] = -1
        
    if return_indices:
        return bb_patch, (bb_xmin, bb_xmax, bb_ymin, bb_ymax)
    else:
        return bb_patch

def blur_mask(mask, ks=None):
    mask_p = mask.astype(float)
    if ks is None:
        ks = int(min(mask_p.shape)/2.)+1
    mask_win = cv2.GaussianBlur(mask_p, (ks, ks), 0)
    return mask_win

def draw_ellipse(x0, y0, sz_x, sz_y, theta, color='b', ax=None):
    if ax is None:
        f, ax = pl.subplots()
    ax.plot(x0, y0, 'b*')
    ell = Ellipse((x0, y0), sz_x, sz_y, angle=np.rad2deg(theta))
    ell.set_alpha(0.7)
    ell.set_edgecolor(color)
    ell.set_facecolor('none')
    ell.set_linewidth(1)
    ax.add_patch(ell) 
    
    return ax

def colorbar(mappable, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    if label is not None:
        cax.set_title(label)
    return cbar


def orig_patch_blurred(im_screen, x0, y0, sz_x, sz_y, rf_theta, ks=101):
    '''
    rf_theta is in degrees
    '''
    curr_rf_theta = np.deg2rad(rf_theta)
    curr_rf_mask, curr_rf_bbox = rf_mask_to_screen(x0, y0, sz_x, sz_y, curr_rf_theta,
                                              resolution=im_screen.shape)
    msk_xmin, msk_xmax, msk_ymin, msk_ymax = curr_rf_bbox
    rf_mask_patch = curr_rf_mask[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    # Bitwise AND operation to black out regions outside the mask
    result = im_screen * curr_rf_mask
    rf_patch = result[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    # blur RF edges
    blurred_mask = blur_mask(curr_rf_mask, ks=ks)
    win_patch = blurred_mask*im_screen

    # crop bbox
    win_patch = win_patch[msk_xmin:msk_xmax, msk_ymin:msk_ymax]

    return result, rf_patch, win_patch


# RF masking

def get_RF_mask_on_screen(rid, curr_rfs, sim_screen,size_name='fwhm'):

    #x0_deg, y0_deg, fwhm_x_deg, fwhm_y_deg, std_x_deg, std_y_deg = curr_rfs.loc[rid][convert_params].values

    rfparams_deg = get_params_for_ellipse(rid, curr_rfs, size_name=size_name)
    rfparams_pix = params_deg_to_pixels(rfparams_deg)
    x0, y0, sz_x, sz_y, theta = rfparams_pix

    mask = np.zeros_like(sim_screen.astype(np.uint8))
    
    # create a white filled ellipse
    mask=cv2.ellipse(mask, (int(x0), int(y0)), (int(sz_x)/2, int(sz_y)/2), 
                     np.rad2deg(theta), 
                     startAngle=360, endAngle=0, color=1, thickness=-1)
    return mask

def get_mask_patch(mask, replace_nans=False):
    '''
    Return mask PATCH
    
    '''
    mask_nan = mask.copy().astype(float)
    mask_nan[mask==0] = np.nan
    mask_bb, bbox_lims = get_bbox_around_nans(mask_nan, replace_nans=replace_nans, 
                                            return_indices=True)
    return mask_bb, bbox_lims


def rf_mask_to_screen(x0, y0, fwhm_x, fwhm_y, theta, resolution=[1080, 1920]):
    '''
    Return mask on screen (pixels), and bbox around mask
    
    theta is in RAD.
    
    '''
    # Create mask
    curr_rf_theta = theta #np.deg2rad(theta) #np.pi/2.
    curr_rf_mask = np.zeros(resolution).astype(np.uint8) #_like(im_screen.astype(np.uint8))
    curr_rf_mask=cv2.ellipse(curr_rf_mask, (int(x0), int(y0)), 
                     (int(fwhm_x)/2, int(fwhm_y)/2), 
                     np.rad2deg(curr_rf_theta), 
                     startAngle=360, endAngle=0, color=1, thickness=-1)

    mask_nan = curr_rf_mask.copy().astype(float)
    mask_nan[curr_rf_mask==0]=np.nan

    mask_bb, curr_rf_bbox = get_bbox_around_nans(mask_nan, 
                                                 replace_nans=False, return_indices=True)
    return curr_rf_mask, curr_rf_bbox


def get_params_for_ellipse(rid, curr_rfs, size_name='fwhm'):
    # convert_params=['x0', 'y0', 'fwhm_x', 'fwhm_y', 'std_x', 'std_y', 'theta']
    #x0_deg, y0_deg, fwhm_x_deg, fwhm_y_deg, std_x_deg, std_y_deg, theta 

    convert_params=['x0', 'y0', '%s_x' % size_name, '%s_y' % size_name, 'theta']
    
    #return (x0_deg, y0_deg, fwhm_x_deg, fwhm_y_deg, theat) #, std_x_deg, std_y_deg
    return (curr_rfs.loc[rid][convert_params].values)

def params_deg_to_pixels((ctx, cty, sigx, sigy, theta), 
                         pix_per_deg=16.06, resolution=[1080, 1920]):
    '''
    RF fit params in degrees, convert to pixel space for drawing
    '''
    lin_x, lin_y = putils.get_lin_coords(resolution=resolution)

    # Get position

    # Convert to deg
    _, yy = np.where(abs(lin_x-ctx)==abs(lin_x-ctx).min())
    x0=int(np.unique(yy))
    xx, _ = np.where(abs(lin_y-cty)==abs(lin_y-cty).min())
    y0=resolution[0]-int(np.unique(xx))

    # Get sigmax-y:
    sz_x = sigx*pix_per_deg #*.5
    sz_y = sigy*pix_per_deg #*.5
    
    return (x0, y0, sz_x, sz_y, theta)



# FFT
import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy import fftpack

def get_fft_magnitude(img):
    '''
    https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
    '''
    M, N = img.shape
    F = fftpack.fftn(img)
    F_magnitude = np.abs(F)
    F_magnitude = fftpack.fftshift(F_magnitude)
    
    return F_magnitude, M, N

def plot_psd_2d(F_magnitude, M, N, cmap='viridis', ax=None):
    if ax is None:
        f, ax = pl.subplots()

    im=ax.imshow(np.log(1 + F_magnitude), cmap=cmap,
              extent=(-N // 2, N // 2, -M // 2, M // 2))
    # ax.set_title('Spectrum magnitude');
    ax.set_aspect('equal')
    colorbar(im)
    return ax


def makeSpectrum(E, dx=None, dy=None, upsample=10):
    """
    https://stackoverflow.com/questions/45496634/two-dimensional-fft-showing-unexpected-frequencies-above-nyquisit-limit
    
    Convert a time-domain array `E` to the frequency domain via 2D FFT. `dx` and
    `dy` are sample spacing in x (left-right, 1st axis) and y (up-down, 0th
    axis) directions. An optional `upsample > 1` will zero-pad `E` to obtain an
    upsampled spectrum.

    Returns `(spectrum, xf, yf)` where `spectrum` contains the 2D FFT of `E`. If
    `Ny, Nx = spectrum.shape`, `xf` and `yf` will be vectors of length `Nx` and
    `Ny` respectively, containing the frequencies corresponding to each pixel of
    `spectrum`.

    The returned spectrum is zero-centered (via `fftshift`). The 2D FFT, and
    this function, assume your input `E` has its origin at the top-left of the
    array. If this is not the case, i.e., your input `E`'s origin is translated
    away from the first pixel, the returned `spectrum`'s phase will *not* match
    what you expect, since a translation in the time domain is a modulation of
    the frequency domain. (If you don't care about the spectrum's phase, i.e.,
    only magnitude, then you can ignore all these origin issues.)
    """
    dx = E.shape[0] if dx is None else dx
    dy = E.shape[1] if dy is None else dy
    
    zeropadded = np.array(E.shape) * upsample
    F = fft.fftshift(fft.fft2(E, zeropadded)) / E.size
    xf = fft.fftshift(fft.fftfreq(zeropadded[1], d=dx))
    yf = fft.fftshift(fft.fftfreq(zeropadded[0], d=dy))
    return (F, xf, yf)

def extents(f):
    "Convert a vector into the 2-element extents vector imshow needs"
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

def plotSpectrum(F, xf, yf, logplot=True, ax=None, label_axes=True, cmap='gray'):
    "Plot a spectrum array and vectors of x and y frequency spacings"
    mag = np.abs(F)
    if logplot:
        psd = 10*np.log10(mag**2)
        label = 'dB'
    else:
        psd = mag.copy()
        label = 'magnitude'
    if ax is None:
        f, ax = pl.subplots() #pl.figure()
    im = ax.imshow(psd,
               aspect="equal",
               interpolation="none",
               origin="lower",
               extent=extents(xf) + extents(yf), 
               cmap=cmap)
    colorbar(im, label=label)
    if label_axes:
        ax.set_xlabel('f_x (Hz)')
        ax.set_ylabel('f_y (Hz)')
        ax.set_title('Power Spectrum')
    #pl.show()
    return ax


# # Sinusoid frequency, in Hz
# x0 = 1.9
# y0 = -2.9

# # Generate data
# im = np.exp(2j * np.pi * (y[:, np.newaxis] * y0 + x[np.newaxis, :] * x0))

# Generate spectrum and plot
# spectrum, xf, yf = makeSpectrum(result, res[1], res[0], upsample=1) #x[1] - x[0], y[1] - y[0])
# plotSpectrum(spectrum, xf, yf)

# # Report peak
# #peak = spectrum[:, np.isclose(xf, x0)][np.isclose(yf, y0)]
# #peak = peak[0, 0]
# #print('spectral peak={}'.format(peak))

def gabor_patch(size, sf=None, lambda_=None, theta=90, sigma=None, 
                phase=0, trim=.005, pix_per_degree=16.05,return_grating=False):
    """Create a Gabor Patch

    size : int
        Image size (n x n)

    lambda_ : int
        Spatial frequency (px per cycle)

    theta : int or float
        Grating orientation in degrees

    sigma : int or float
        gaussian standard deviation (in pixels)

    phase : float
        0 to 1 inclusive
    """
    assert not (sf is None and lambda_ is None), "Must specify sf or lambda)_"
    
    deg_per_pixel=1./pix_per_degree
    lambda_ = 1./(sf*deg_per_pixel) # cyc per pixel (want: pix/cyc)
    
    if sigma is None:
        sigma = max(size)
    
    sz_y, sz_x = size
    
    # make linear ramp
    X0 = (np.linspace(1, sz_x, sz_x) / sz_x) - .5
    Y0 = (np.linspace(1, sz_y, sz_y) / sz_y) - .5

    # Set wavelength and phase
    freq = sz_x / float(lambda_)
    phaseRad = phase * 2 * np.pi

    # Make 2D grating
    Ym, Xm = np.meshgrid(X0, Y0)

    # Change orientation by adding Xm and Ym together in different proportions
    thetaRad = (theta / 360.) * 2 * np.pi
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    grating = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad)

    # 2D Gaussian distribution
    gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(sz_x)) ** 2))

    # Trim
    gauss[gauss < trim] = 0

   
    if return_grating:
        print("returning grating")
        return grating
    else: 
        return grating * gauss


#===================================================================
# FFT functions 
#===================================================================

def get_fft(img, map_type='mag'):
    M, N = img.shape
    y = np.fft.fft2(img)
    if map_type=='mag':
        pwr = np.abs(y)
    elif map_type=='power':
        pwr = np.abs(y)**2
    else:
        pwr = 10*np.log10(np.abs(y)**2)
    return pwr, M, N

def get_psd_2D(img, map_type='mag', shift=True):
    fft_result, M, N = get_fft(img, map_type=map_type)
    
    if shift:
        fft_result = np.fft.fftshift(fft_result)
        
    return fft_result, M, N



def get_psd_1D(psd2D, img=None, average=False, cyc_per_deg=False, pix_per_deg=16.05):
    h  = psd2D.shape[0] # smaller, y-axis
    w  = psd2D.shape[1] # x-axis (bigger)
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    
    if cyc_per_deg:
        assert img is not None, "Provide original img"
        xf, yf = get_freqs(img, cyc_per_deg=False, pix_per_deg=pix_per_deg)
        Y, X = np.meshgrid(xf, yf)
        r_freq    = np.hypot(X, Y) #.astype(np.int)
        
    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    if average:
        psd1D = ndimage.mean(psd2D, r, index=np.arange(0, wc))
    else:
        psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))

    if cyc_per_deg:
        return psd1D, r, r_freq
    else:
        return psd1D, r

def get_freqs(img, shift=False, pix_per_deg=16.05, cyc_per_deg=False):
    M, N = img.shape #0=rows, 1=cols
    freqsM = np.fft.fftfreq(M) # rows
    freqsN = np.fft.fftfreq(N) # cols
    
    if shift:
        freqsN = np.fft.fftshift(freqsN)
        freqsM = np.fft.fftshift(freqsM)
        
    if cyc_per_deg:
        return freqsN*pix_per_deg, freqsM*pix_per_deg
    else:
        return freqsN, freqsM # N goes w/ x, or cols; M goes w/ y, or rows
    

def get_psd_results(im, map_type='mag', shift=True, cyc_per_deg=True, average=True):

    # Get 2d power spec density
    psd_2d, M, N = get_psd_2D(im, map_type=map_type, shift=shift)
    
    # 1D average
    psd_1d, r_, r_freq_ = get_psd_1D(psd_2d, img=im, average=average, 
                                     cyc_per_deg=cyc_per_deg)
    # Get frequencies
    freqx, freqy = get_freqs(im, cyc_per_deg=cyc_per_deg, shift=True)

    return psd_2d, psd_1d, freqx, freqy

def get_top_freq(psd_1d, freqx, is_half=False, start_ix=0, fr=1):
    if is_half:
        freqs=freqx.copy()
    else:
        N = len(freqx)
        freqs = freqx[int(float(N)//2):]

    idx = np.nanargmax(psd_1d[start_ix:]) #np.argmax(np.abs(y)[1:])
    top_freq = freqs[start_ix:][idx]
    freq_in_hz = abs(top_freq * fr)
    
    return freq_in_hz, freqs


def plot_top_freqs(psd_1d, freqs, stim_freq=None, is_half=False,
                   max_color='r', stim_color='b', marker='o', start_ix=0,
                   n_pts_plot=-1, fr=1, markersize=5, lw=1, ax=None):
    
    if ax is None:
        f, ax = pl.subplots()

    top_freq_hz, freqs = get_top_freq(psd_1d, freqs, is_half=is_half,
                                          start_ix=start_ix, fr=fr)
    
    ax.plot(freqs[start_ix:n_pts_plot]*fr, psd_1d[start_ix:n_pts_plot], 'k-', 
            lw=lw, markersize=markersize)

    ix_max = np.nanargmin(np.abs(freqs - top_freq_hz)) #.nanargmin()
    ax.plot(top_freq_hz, psd_1d[ix_max], color=max_color, 
            marker=marker, label='mx %.2f' % top_freq_hz,
           markersize=markersize)

    if stim_freq is not None:
        ix_sim = np.nanargmin(np.abs(freqs - stim_freq)) #.nanargmin()
        ax.plot(freqs[ix_sim],  psd_1d[ix_sim], color=stim_color, 
                marker=marker, label='sm %.2f' % stim_freq,
                markersize=markersize)
    sns.despine(ax=ax, trim=True, offset=2)
    
    return top_freq_hz, ax

def get_max_across_bins(bins, values):
    
    # 1d avg spectrum
    find_val = np.nanmax(values)
    
    max_ixs = np.where(values==find_val)
    max_theta = np.mean([bins[i] for i in max_ixs])
    max_val_at_theta = np.mean([values[i] for i in max_ixs])
    
    return max_theta, max_val_at_theta


def find_match_across_bins(find_val, bins, values):
    
    # 1d avg spectrum
    close_ixs = np.where(abs(bins-find_val)==abs(bins-find_val).min())
    closest_bin = np.mean([bins[i] for i in close_ixs])
    closest_val = np.mean([values[i] for i in close_ixs])
    
    return closest_bin, closest_val



