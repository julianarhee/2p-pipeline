
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import optparse
import numpy as np
import time
import sys
import os
import glob
# option to import from github folder
#sys.path.insert(0, '/n/coxfs01/cechavarria/repos/suite2p')
import suite2p
from suite2p import fig

class stat_file:
    def __init__(self,fname):
        self.fname = fname
        self.loaded = False
        self.ops_plot = []
        # default plot options
        self.ops_plot.append(True)
        for k in range(6):
            self.ops_plot.append(0)
            
        self.colors = [
            "A: random",
            "S: skew",
            "D: compact",
            "F: footprint",
            "G: aspect_ratio",
            "H: chan2_prob",
            "J: classifier, cell prob=",
            "K: correlations, bin=",
        ]
        self.imerge = [0]
        self.ichosen = 0
        
    def load_proc(self):
        name = self.fname
        print(name)
        try:
            stat = np.load(name)
            ypix = stat[0]["ypix"]
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a stat.npy file :( '
                  '(needs stat[n]["ypix"]!)')
            stat = None
        if stat is not None:
            basename, fname = os.path.split(name)
            goodfolder = True
            try:
                Fcell = np.load(basename + "/F.npy")
                Fneu = np.load(basename + "/Fneu.npy")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print(
                    "ERROR: there are no fluorescence traces in this folder "
                    "(F.npy/Fneu.npy)"
                )
                goodfolder = False
            try:
                Spks = np.load(basename + "/spks.npy")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("there are no spike deconvolved traces in this folder "
                      "(spks.npy)")
                goodfolder = False
            try:
                ops = np.load(basename + "/ops.npy")
                ops = ops.item()
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("ERROR: there is no ops file in this folder (ops.npy)")
                goodfolder = False
            try:
                iscell = np.load(basename + "/iscell.npy")
                probcell = iscell[:, 1]
                iscell = iscell[:, 0].astype(np.bool)
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("no manual labels found (iscell.npy)")
                if goodfolder:
                    NN = Fcell.shape[0]
                    iscell = np.ones((NN,), np.bool)
                    probcell = np.ones((NN,), np.float32)
            try:
                redcell = np.load(basename + "/redcell.npy")
                probredcell = redcell[:,1].copy()
                redcell = redcell[:,0].astype(np.bool)
                self.hasred = True
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("no channel 2 labels found (redcell.npy)")
                self.hasred = False
                if goodfolder:
                    NN = Fcell.shape[0]
                    redcell = np.zeros((NN,), np.bool)
                    probredcell = np.zeros((NN,), np.float32)

            if goodfolder:
                self.basename = basename
                self.stat = stat
                self.ops = ops
                self.Fcell = Fcell
                self.Fneu = Fneu
                self.Spks = Spks
                self.iscell = iscell
                self.probcell = probcell
                self.redcell = redcell
                self.probredcell = probredcell
                for n in range(len(self.stat)):
                    self.stat[n]['chan2_prob'] = self.probredcell[n]
                self.loaded = True
            else:
                print("stat.npy found, but other files not in folder")
                Text = ("stat.npy found, but other files missing, "
                        "choose another?")
                self.load_again(Text)
        else:
            Text = "Incorrect file, not a stat.npy, choose another?"
            self.load_again(Text)
            
    
    def init_all_masks(parent):
        '''creates RGB masks using stat and puts them in M0 or M1 depending on
        whether or not iscell is True for a given ROI
        args:
            ops: mean_image, Vcorr
            stat: xpix,ypix,xext,yext
            iscell: vector with True if ROI is cell
            ops_plot: plotROI, view, color, randcols
        outputs:
            M0: ROIs that are True in iscell
            M1: ROIs that are False in iscell
        '''
        ops = parent.ops
        stat = parent.stat
        iscell = parent.iscell
        cols = parent.ops_plot[3]

        ncells = len(stat)
        Ly = ops['Ly']
        Lx = ops['Lx']
        Sroi  = np.zeros((3,Ly,Lx), np.float32)
        Sext   = np.zeros((3,Ly,Lx), np.float32)
        LamAll = np.zeros((Ly,Lx), np.float32)
        Lam    = np.zeros((3,3,Ly,Lx), np.float32)
        iExt   = -1 * np.ones((3,3,Ly,Lx), np.int32)
        iROI   = -1 * np.ones((3,3,Ly,Lx), np.int32)

        for n in range(ncells-1,-1,-1):
            ypix = stat[n]['ypix']
            if ypix is not None:
                xpix = stat[n]['xpix']
                yext = stat[n]['yext']
                xext = stat[n]['xext']
                lam = stat[n]['lam']
                lam = lam / lam.sum()
                i = int(1-iscell[n])
                # add cell on top
                iROI[i,2,ypix,xpix] = iROI[i,1,ypix,xpix]
                iROI[i,1,ypix,xpix] = iROI[i,0,ypix,xpix]
                iROI[i,0,ypix,xpix] = n
                # add outline to all layers
                iExt[i,2,yext,xext] = iExt[i,1,yext,xext]
                iExt[i,1,yext,xext] = iExt[i,0,yext,xext]
                iunder = iExt[i,1,yext,xext]
                iExt[i,0,yext,xext] = n
                # add weighting to all layers
                Lam[i,2,ypix,xpix] = Lam[i,1,ypix,xpix]
                Lam[i,1,ypix,xpix] = Lam[i,0,ypix,xpix]
                Lam[i,0,ypix,xpix] = lam
                Sroi[i,ypix,xpix] = 1
                Sext[i,yext,xext] = 1
                LamAll[ypix,xpix] = lam
                
                #repeat to get all cells in same figure
                # add cell on top
                iROI[2,2,ypix,xpix] = iROI[2,1,ypix,xpix]
                iROI[2,1,ypix,xpix] = iROI[2,0,ypix,xpix]
                iROI[2,0,ypix,xpix] = n
                # add outline to all layers
                iExt[2,2,yext,xext] = iExt[2,1,yext,xext]
                iExt[2,1,yext,xext] = iExt[2,0,yext,xext]
                iunder = iExt[2,1,yext,xext]
                iExt[2,0,yext,xext] = n
                # add weighting to all layers
                Lam[2,2,ypix,xpix] = Lam[2,1,ypix,xpix]
                Lam[2,1,ypix,xpix] = Lam[2,0,ypix,xpix]
                Lam[2,0,ypix,xpix] = lam
                Sroi[2,ypix,xpix] = 1
                Sext[2,yext,xext] = 1
                LamAll[ypix,xpix] = lam

        LamMean = LamAll[LamAll>1e-10].mean()
        RGBall = np.zeros((3,cols.shape[1]+1,6,Ly,Lx,3), np.float32)
        Vback   = np.zeros((5,Ly,Lx), np.float32)
        RGBback = np.zeros((5,Ly,Lx,3), np.float32)

        for k in range(6):
            if k>0:
                if k==2:
                    if 'meanImgE' not in ops:
                        ops = utils.enhanced_mean_image(ops)
                    mimg = ops['meanImgE']
                elif k==1:
                    mimg = ops['meanImg']
                    S = np.maximum(0,np.minimum(1, Vorig*1.5))
                    mimg1 = np.percentile(mimg,1)
                    mimg99 = np.percentile(mimg,99)
                    mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                    mimg = np.maximum(0,np.minimum(1,mimg))
                elif k==3:
                    vcorr = ops['Vcorr']
                    mimg1 = np.percentile(vcorr,1)
                    mimg99 = np.percentile(vcorr,99)
                    vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
                    mimg = mimg1 * np.ones((ops['Ly'],ops['Lx']),np.float32)
                    mimg[ops['yrange'][0]:ops['yrange'][1],
                        ops['xrange'][0]:ops['xrange'][1]] = vcorr
                    mimg = np.maximum(0,np.minimum(1,mimg))
                elif k==4:
                    if 'meanImg_chan2_corrected' in ops:
                        mimg = ops['meanImg_chan2_corrected']
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                        mimg = np.maximum(0,np.minimum(1,mimg))
                elif k==5:
                    if 'meanImg_chan2' in ops:
                        mimg = ops['meanImg_chan2']
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                        mimg = np.maximum(0,np.minimum(1,mimg))
                else:
                    mimg = np.zeros((ops['Ly'],ops['Lx']),np.float32)

                Vback[k-1,:,:] = mimg
                V = mimg
                V = np.expand_dims(V,axis=2)
            for i in range(3):
                Vorig = np.maximum(0, np.minimum(1, 0.75*Lam[i,0,:,:]/LamMean))
                Vorig = np.expand_dims(Vorig,axis=2)
                if k==3:
                    S = np.expand_dims(Sext[i,:,:],axis=2)
                    Va = np.maximum(0,np.minimum(1, V + S))
                else:
                    S = np.expand_dims(Sroi[i,:,:],axis=2)
                    if k>0:
                        S     = np.maximum(0,np.minimum(1, Vorig*1.5))
                        Va    = V
                    else:
                        Va = Vorig
                for c in range(0,cols.shape[1]):
                    if k==3:
                        H = cols[iExt[i,0,:,:],c]
                        H = np.expand_dims(H,axis=2)
                        hsv = np.concatenate((H,S,Va),axis=2)
                        RGBall[i,c,k,:,:,:] = fig.hsv_to_rgb(hsv)
                    else:
                        H = cols[iROI[i,0,:,:],c]
                        H = np.expand_dims(H,axis=2)
                        hsv = np.concatenate((H,S,Va),axis=2)
                        RGBall[i,c,k,:,:,:] = fig.hsv_to_rgb(hsv)

        for k in range(5):
            H = np.zeros((Ly,Lx,1),np.float32)
            S = np.zeros((Ly,Lx,1),np.float32)
            V = np.expand_dims(Vback[k,:,:],axis=2)
            hsv = np.concatenate((H,S,V),axis=2)
            RGBback[k,:,:,:] = fig.hsv_to_rgb(hsv)

        parent.RGBall = RGBall
        parent.RGBback = RGBback
        parent.Vback = Vback
        parent.iROI = iROI
        parent.iExt = iExt
        parent.Sroi = Sroi
        parent.Sext = Sext
        parent.Lam  = Lam
        parent.LamMean = LamMean
        
    def make_mask_figures(self):
      #  self.ops_plot[1] = 0
     #   self.ops_plot[2] = 0
        self.ops_plot[3] = []
        self.ops_plot[4] = []
        self.ops_plot[5] = []
        self.ops_plot[6] = []

        ncells = len(self.stat)
        for n in range(0, ncells):
            ypix = self.stat[n]["ypix"].flatten()
            xpix = self.stat[n]["xpix"].flatten()
            iext = fig.boundary(ypix, xpix)
            self.stat[n]["yext"] = ypix[iext]
            self.stat[n]["xext"] = xpix[iext]
            ycirc, xcirc = fig.circle(
                self.stat[n]["med"],
                self.stat[n]["radius"]
            )
            goodi = (
                (ycirc >= 0)
                & (xcirc >= 0)
                & (ycirc < self.ops["Ly"])
                & (xcirc < self.ops["Lx"])
            )
            self.stat[n]["ycirc"] = ycirc[goodi]
            self.stat[n]["xcirc"] = xcirc[goodi]

#         # make color arrays for various views
        fig.make_colors(self)
#         # colorbar
        self.colormat = fig.make_colorbar()
        self.init_all_masks()



def make_roi_figures(options):

    #unpack options
    rootdir = options.rootdir
    animalid = options.animalid
    session = options.session
    acquisition = options.acquisition
    run = options.run

    analysis_header = options.analysis

    #figure out directory paths
    data_dir = os.path.join(rootdir,animalid,session,acquisition,run)
    analysis_dir = os.path.join(data_dir,'processed',analysis_header)
    output_dir = os.path.join(analysis_dir,'suite2p','plane0')

    fig_dir = os.path.join(output_dir,'figures')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir) 
    print('Output Directory:%s'%(fig_dir))


    statfn = os.path.join(output_dir,'stat.npy')
    stat1 = stat_file(statfn)
    stat1.load_proc()
    stat1.make_mask_figures()


    view_labels = ['blank','mean_image',
                   'enhanced_image',
                   'correlation_image']

    colormap_labels = [
                "random",
                "skew",
                "compact",
                "footprint",
                "aspect_ratio",
                "chan2_prob",
                "classifier",
                "correlations",
            ]

    ROIgroup_labels = ['cells',
                     'not_cells','all']

    for ROIgroup in range(0,3):
        for view in range(0,2):
            for color in [0,6]:

                fig_fn = '%s_%s_%s.png'%(ROIgroup_labels[ROIgroup],view_labels[view],colormap_labels[color])
                M = np.array(stat1.RGBall[ROIgroup,color,view,:,:,:])


                dpi = 80
                szY,szX,nCh = M.shape
                # What size does the figure need to be in inches to fit the image?
                figsize = szX / float(dpi), szY / float(dpi)

                # Create a figure of the right size with one axes that takes up the full figure
                f = plt.figure(figsize=figsize)
                ax = f.add_axes([0, 0, 1, 1])

                # Hide spines, ticks, etc.
                ax.axis('off')

                ax.imshow(M)
                f.savefig(os.path.join(fig_dir,fig_fn), dpi=dpi, transparent=True)
                plt.close()


class Struct():
    pass

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', help='source dir (root project dir containing all expts) [default: /n/coxfs01/2p-data]')
    parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
    parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID') 
    parser.add_option('-A', '--acq', action='store', dest='acquisition', default='', help="acquisition folder (ex: 'FOV1_zoom3x')")
    parser.add_option('-R', '--run', action='store', dest='run', default='', help='name of run to process') 
    parser.add_option('-Y', '--analysis', action='store', dest='analysis', default='', help='Analysis to process. [ex: suite2p_analysis001]')


    (options, args) = parser.parse_args() 

    return options




#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):
    
    print('Making Custom Figures')
    options = extract_options(options)
    make_roi_figures(options)
    

    
#%%

if __name__ == '__main__':
    main(sys.argv[1:])
