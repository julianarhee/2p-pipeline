#!/usr/bin/env python2
'''
write stuff
'''


import matlab.engine
eng = matlab.engine.start_matlab()

# add paths:
eng.addpath(genpath('/home/juliana/Repositories/ca_source_extraction'))
eng.addpath(genpath('/home/juliana/Repositories/2p-tester-scripts'))


opts = {\
    'source': '/nas/volume1/2photon/RESDATA/TEFO',\
    'session': '20161218_CE024',\
    'run': 'retinotopy5',\
    'datastruct': 100,\
    'acquisition': 'fov2_bar5',\
    'datapath': 'DATA',\
    'tefo': 1,\
    'preprocessing': 'raw',\
    'corrected': 0,\
    'meta': 'SI',\
    'channels': 2,\
    'signalchannel': 1,\
    'roitype': '3Dcnmf',\
    'seedrois': 1,\
    'maskpath': '',\
    'maskdims': '3D',\
    'maskshape': '3Dcontours',\
    'maskfinder': '',\
    'slices': '[1:12]',\
    'averaged': 0,\
    'matchedtiffs': '[]',\
    'excludedtiffs': '[]',\
    'metaonly': 0,\
    'nmetatiffs': 4}

dsoptions = eng.DSoptions(opts)

# D = eng.create_datastruct(dsoptions, 'true') # add flag to overwrite automatically 

eng.quit()
