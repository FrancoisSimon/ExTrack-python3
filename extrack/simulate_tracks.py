#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:16:58 2021

@author: mcm
"""
import numpy as np

def markovian_process(TrMat, # transition matrix e.g. np.array([[1-p01,p01], [p10,1-p10]])
                      initial_fractions = [0.5,0.5], # fraction in each states at the beginning of the tracks
                      track_len = 10, # nb of positions per track
                      nb_tracks = 10000 # number of tracks
                      ) :
    '''
    outputs a 2D array of markovian states (hidden) of nb_tracks tracks row and track_len cols 
    '''

    nb_states = len(TrMat)
    cumMat = np.cumsum(TrMat, 1)
    cumF = np.cumsum(initial_fractions)
    states = np.zeros((nb_tracks, track_len)).astype(int)
    randoms = np.random.rand(nb_tracks, track_len)
    for s in range(nb_states -1):
        states[:,0] += (randoms[:,0]>cumF[s]).astype(int)
    for k in range(1,track_len):
        for s in range(nb_states -1):
            states[:,k] += (randoms[:,k] > cumMat[states[:,k-1]][:,s]).astype(int)
    return states

def get_fractions_from_TrMat(TrMat):
    if len(TrMat) == 2:
        p01 = TrMat[0,1]
        p10 = TrMat[1,0]
        initial_fractions = np.array([p10 / (p10+p01), p01 / (p10+p01)])
    elif len(TrMat) == 3:
        p01 = TrMat[0,1]
        p02 = TrMat[0,2]
        p10 = TrMat[1,0]
        p12 = TrMat[1,2]
        p20 = TrMat[2,0]
        p21 = TrMat[2,1]
        F0 = (p10*(p21+p20)+p20*p12)/((p01)*(p12 + p21) + p02*(p10 + p12 + p21) + p01*p20 + p21*p10 + p20*(p10+p12))
        F1 = (F0*p01 + (1-F0)*p21)/(p10 + p12 + p21)
        initial_fractions =  np.array([F0, F1, 1-F0-F1])
    else:
        '''
        if more states lets just run the transition process until equilibrium
        '''
        A0 = np.ones(len(TrMat))/len(TrMat)
        k = 0
        prev_A = A0
        A = np.dot(A0, TrMat)
        while not np.all(prev_A == A):
            k += 1
            prev_A = A
            A = np.dot(A, TrMat)
            if k > 10000000:
                raise Exception("We could't find a steady state, convergence didn't occur within %s steps, \ncurrent fractions : %s"%(k, A))
        initial_fractions = A
    return initial_fractions

def get_tracks(track_lengths = [7,8,9,10,11], # create arrays of tracks of specified number of localizations  
               track_nb_dist = [1000, 800, 700, 600, 550], # list of number of tracks per array
               LocErr = 0.02, # Localization error in um
               Ds = [0, 0.05], # diffusion coef of each state in um^2/s
               TrMat = np.array([[0.9,0.1], [0.2,0.8]]), # transition matrix e.g. np.array([[1-p01,p01], [p10,1-p10]])
               initial_fractions = None, # fraction in each states at the beginning of the tracks, if None assums steady state determined by TrMat
               dt = 0.02, # time in between positions in s
               nb_dims = 2 # number of spatial dimensions : 2 for (x,y) and 3 for (x,y,z)
               ):
    '''
    create tracks with specified kinetics
    outputs the tracks and the actual hidden states using the ExTrack format (dict of np.arrays)
    any states number 'n' can be producted as long as len(Ds) = len(initial_fractions) and TrMat.shape = (n,n)
    '''
    
    Ds = np.array(Ds)
    TrMat = np.array(TrMat)
    nb_sub_steps = 30
    
    if initial_fractions is None:
        initial_fractions = get_fractions_from_TrMat(TrMat)
        
    nb_states = len(TrMat)
    sub_dt = dt/nb_sub_steps
    
    TrSubMat = TrMat / nb_sub_steps
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 0
    TrSubMat[np.arange(nb_states),np.arange(nb_states)] = 1 - np.sum(TrSubMat,1)
    
    all_Css = []
    all_Bss = []
    
    for nb_tracks, track_len in zip(track_nb_dist, track_lengths):
        print(nb_tracks, track_len)
        
        states = markovian_process(TrSubMat,
                                   initial_fractions = initial_fractions, 
                                   track_len =(track_len-1) * nb_sub_steps + 1,
                                   nb_tracks = nb_tracks)
        # determines displacements then MSDs from stats
        positions = np.random.normal(0, 1, (nb_tracks, (track_len-1) * nb_sub_steps+1, nb_dims)) * np.sqrt(2*Ds*sub_dt)[states[:,:, None]]
        positions = np.cumsum(positions, 1)

        positions = positions + np.random.normal(0, LocErr, (nb_tracks, (track_len-1) * nb_sub_steps + 1, nb_dims))
        
        all_Css.append(positions[:,np.arange(0,(track_len-1) * nb_sub_steps +1, nb_sub_steps)])
        all_Bss.append(states[:,np.arange(0,(track_len-1) * nb_sub_steps +1, nb_sub_steps)])
    
    all_Css_dict = {}
    all_Bss_dict = {}
    for Cs, Bs in zip(all_Css, all_Bss):
        l = str(Cs.shape[1])
        all_Css_dict[l] = Cs
        all_Bss_dict[l] = Bs
        
    return all_Css_dict, all_Bss_dict

import os
os.chdir('C:/Users/francois/Documents/ExTrack-python3-main/ExTrack-python3-main')

exec('extrack/__init__.py')

track_lengths = [7,8,9,10,11] # create arrays of tracks of specified number of localizations  
track_nb_dist = [1000, 800, 700, 600, 550] # list of number of tracks per array
LocErr = 0.02 # Localization error in um
Ds = [0, 0.1] # diffusion coef of each state in um^2/s
TrMat = np.array([[0.9,0.1], [0.2,0.8]]) # transition matrix e.g. np.array([[1-p01,p01], [p10,1-p10]])
initial_fractions = None # fraction in each states at the beginning of the tracks, if None assums steady state determined by TrMat
dt = 0.02 # time in between positions in s
nb_dims = 2 # number of spatial dimensions : 2 for (x,y) and 3 for (x,y,z)

all_Css, all_Bss = get_tracks(track_lengths = track_lengths,
                              track_nb_dist = track_nb_dist,
                              LocErr = LocErr,
                              Ds = Ds,
                              TrMat = TrMat,
                              initial_fractions = initial_fractions, 
                              dt = dt,
                              nb_dims = nb_dims)

model_fit, preds = auto_fitting_2states(all_Css,dt)

df_DATA = extrack_to_pandas(all_Css, frames = None, opt_metrics = {}, pred_Bs = preds)

df_DATA.to_csv('C:/Users/francois/Documents/tracks.csv')

import xml.etree.cElementTree as ElementTree

def write_trackmate(da):
    """Experimental (not to say deprecated)"""
    #tree = ElementTree.Element('tmx', {'version': '1.4a'})
    Tracks = ElementTree.Element('Tracks', {'lol': 'oui'})
    ElementTree.SubElement(Tracks,'header',{'adminlang': 'EN',})
    ElementTree.SubElement(Tracks,'body')
    
    with open('C:/Users/francois/Documents/tracks.xml', 'wb') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>')
        ElementTree.ElementTree(Tracks).write(f, 'utf-8')





path = 'C:/Users/francois/Documents/tracks.xml'

def save_df_2_xml(df_DATA, path, dt)
    track_IDs = np.unique(df_DATA['track_ID'])
    with open(path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<Tracks nTracks="%s" spaceUnits="µm" frameInterval="%s" timeUnits="ms" generationDateTime="lun., 14 déc. 2020 11:25:17" from="TrackMate v4.0.1">\n'%(len(track_IDs), dt))
        preds_str = ''
        remove()
        colnames = df_DATA.columns.to_list()
        colnames.remove('X')
        colnames.remove('Y')
        colnames.remove('frame')
        colnames.remove('track_ID')
        pred_cols = []
        [pred_cols.append(element) for element in colnames if element.startswith('pred_')]
        other_cols = []
        [other_cols.append(element) for element in colnames if not element.startswith('pred_')]

        for track_ID in track_IDs:
            track = df_DATA[df_DATA['track_ID']==0]
            f.write('  <particle nSpots="%s">\n'%(len(track.index)))
            for idx in track.index:
                pos = track.iloc[idx]
                f.write('    <detection t="%s" x="%s" y="%s" z="%s" />\n'%(pos['frame'],pos['X'],pos['Y'],0.0))

                print(row)
                dir(track)
                track.index
df_DATA.to_xml('C:/Users/francois/Documents/tracks.xml', parser = 'etree')


type('<?xml version="1.0" encoding="UTF-8" ?>')

track_lengths = [9,11]
track_nb_dist = [100,200]


track_lengths = [7,8,9,10,11] # create arrays of tracks of specified number of localizations  
track_nb_dist = [1000, 800, 700, 600, 550] # list of number of tracks per array

track_lengths = [11]
track_nb_dist = [200]
LocErr = 0.02 # Localization error in um
Ds = [0, 0.02, 0.1] # diffusion coef of each state in um^2/s
TrMat = np.array([[0.85,0.05,0.1],[0.1,0.8,0.1],[0.03,0.03,0.94]])
initial_fractions = None
dt = 0.02 # time in between positions in s
nb_dims = 2 # number of spatial dimensions : 2 for (x,y) and 3 for (x,y,z)

all_Css, all_Bss = get_tracks(track_lengths = track_lengths,
                              track_nb_dist = track_nb_dist,
                              LocErr = LocErr,
                              Ds = Ds,
                              TrMat = TrMat,
                              initial_fractions = initial_fractions, 
                              dt = dt,
                              nb_dims = nb_dims)

model_fit, preds = auto_fitting_3states(all_Css,dt,estimated_vals = { 'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.022, 'D2' :  0.15, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1})
42252.36

df_DATA = extrack_to_pandas(tracks, frames, opt_metrics, pred_Bs)

import xml.etree.cElementTree as ElementTree

def write_trackmate(da):
    """Experimental (not to say deprecated)"""
    #tree = ElementTree.Element('tmx', {'version': '1.4a'})
    Tracks = ElementTree.Element('Tracks', {'lol': 'oui'})
    ElementTree.SubElement(Tracks,'header',{'adminlang': 'EN',})
    ElementTree.SubElement(Tracks,'body')

    with open('/home/**/Bureau/myfile.xml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>')
        ElementTree.ElementTree(Tracks).write(f, 'utf-8')




all_Css, all_Bss = get_tracks(track_lengths = track_lengths, track_nb_dist = track_nb_dist, initial_fractions = None)
all_Css, all_Bss = get_tracks(track_lengths = track_lengths, track_nb_dist = track_nb_dist, initial_fractions = [0.2,0.8])
all_Css, all_Bss = get_tracks(track_lengths, track_nb_dist, LocErr, Ds, TrMat, initial_fractions = initial_fractions, dt = dt)

track_nb_dist = [20000]
track_lengths = [20]
LocErr = 0.02
TrMat = np.array([[0.77,0.11,0.12],[0.13,0.73,0.14],[0.15,0.16,0.69]])
TrMat = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
TrMat = np.array([[0.85,0.05,0.1],[0.1,0.8,0.1],[0.03,0.03,0.94]])
initial_fractions = np.array([0.13,0.07,0.8])
Ds = np.array([1e-10,0.02, 0.08])
dt = 0.06

np.mean(all_Bss['20'] == 0, 0)
np.mean(all_Bss['20'] == 1, 0)
np.mean(all_Bss['20'] == 2, 0)

all_Css, all_Bss = get_multi_tracks_multi_states(track_nb_dist, track_lengths, LocErr, Ds, initial_fractions, TrMat, dt)

fit = get_2DSPT_params(all_Css, dt, nb_substeps = 1, states_nb = 3, frame_len = 4, verbose = 1, method = 'powell', vary_params = [True, False, True, True, True, True,True, True, True, True,True, True],estimated_vals = [0.02, 1e-12, 0.02, 0.1, 0.2, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1],min_values = [0.007, 1e-12, 0.0001, 0.001, 0.001,0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], max_values = [0.6,1,1,1,1,1,0.4,0.4,0.4,0.4,0.4,0.4])

fit = get_2DSPT_params(all_Css, dt, nb_substeps = 1, states_nb = 2, frame_len = 4, verbose = 1, method = 'powell')

82462


all_Bss[0][-10:]







from glob import glob
import xmltodict
import numpy as np
from matplotlib import pyplot as plt

def read_multi_trackmate_xml(path, lengths=[6,7,8,9,10,11,12,13,14,15], dist_th = 0.3):
    """Converts xml to [x, y, time, frame] table"""
    data = xmltodict.parse(open(path, 'r').read(), encoding='utf-8')
    # Checks
    spaceunit = data['Tracks']['@spaceUnits']
    if spaceunit not in ('micron', 'um', 'µm', 'Âµm'):
        raise IOError("Spatial unit not recognized: {}".format(spaceunit))
    if data['Tracks']['@timeUnits'] != 'ms':
        raise IOError("Time unit not recognized")

    # parameters
    framerate = float(data['Tracks']['@frameInterval'])/1000. # framerate in ms
    traces = {}
    frames = {}
    for l in lengths:
        traces[str(l)] = []
        frames[str(l)] = []
    try:
        for i, particle in enumerate(data['Tracks']['particle']):
            track = [(float(d['@x']), float(d['@y']), float(d['@t'])*framerate, int(d['@t']), i) for d in particle['detection']]
            #print(len(track))
            track = np.array(track)
            no_zero_disp = np.min((track[1:,0] - track[:-1,0])**2) * np.min((track[1:,1] - track[:-1,1])**2)
            no_zero_disp = no_zero_disp > 0
            dists = np.sum((track[1:, :2] - track[:-1, :2])**2, axis = 1)**0.5
            if no_zero_disp and track[0, 3] > 400 and np.all(dists<dist_th):
                if np.any([len(track)]*len(lengths) == np.array(lengths)) :
                    l = np.min([len(track), np.max(lengths)])
                    traces[str(l)].append(track[:, 0:2])
                    frames[str(l)].append(track[:, 3])
                elif len(track) > np.max(lengths):
                    traces[str(np.max(lengths))].append(track[:np.max(lengths), 0:2])
                    frames[str(np.max(lengths))].append(track[:np.max(lengths), 3])
    except KeyError as e:
        print('problem with {path}')
        raise e
    '''traces_list = []
    frames_list = []
    for l in lengths:
        if len(traces[str(l)])>0:
            traces_list.append(np.array(traces[str(l)]))
            frames_list.append(np.array(frames[str(l)]))'''
    return traces, frames

ex = '2020-12-14/Tracks/*-*'
exps = glob('/run/user/1000/gvfs/smb-share:server=gaia.pasteur.fr,share=qbact/Francois/' + ex)
exps = np.sort(exps)
dts = [0.06]*20

ex = 'PBP1b-AV51/Tracks/*-*'
exps = glob('/run/user/1000/gvfs/smb-share:server=gaia.pasteur.fr,share=qbact/Francois/' + ex)
exps = np.sort(exps)
dts = [0.06]*20



F0s = []
p01s = []
p10s = []
D0s = []
D1s = []
LocErrs = []
LocErr = 0.027
Fs = [0.65,0.35]
exp = exps[0]  # exps[1] for 2020-10-08/Tracks/RodZtest-40
fitted_ds = []
all_Ds = []
MSDs = []
all_lens = []
dt =0.06
n=1

fit_params = []

for i,exp in enumerate(exps):
    dt=dts[i]
    print(exp, dt)

    movies = glob(exp + '/*.xml')
    lengths = np.arange(5,15)
    #for min_len in [5,7,9]:#[5,7,9,13]: #[3,4,5,7,8,10]
        #lengths = [min_len]
    data = {}
    for l in lengths:
        data[str(l)] = np.array([[[]]*l]*2).T
    
    for movie in movies:
        cur_data, frames = read_multi_trackmate_xml(movie, lengths, dist_th = 0.3)
        for l in lengths:
            if len(cur_data[str(l)])>0:
                data[str(l)] = np.concatenate((data[str(l)], cur_data[str(l)]))
    all_len = []
    for l in lengths:
        all_len.append(len(data[str(l)]))
    print(all_len)
    all_lens.append(all_len)

    nb_substeps = 1
    data_list = []
    for l in lengths:
        if len(data[str(l)])>0:
            data_list.append(np.array(data[str(l)][:,:])[:,np.arange(0,l,n)])
    
    model_fit = get_2DSPT_params(data_list, dt = n*dt,states_nb =3, nb_substeps = 1, verbose = 1, frame_len = 5, steady_state = False, vary_params = [True, False, True, True, True, True,True, True, True, True,True, True],estimated_vals = [0.02, 1e-12, 0.02, 0.1, 0.2, 0.1, 0.1,0.1,0.1,0.1,0.05,0.05],min_values = [0.007, 1e-12, 0.0001, 0.001, 0.001,0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], max_values = [0.6,1,1,1,1,1,0.4,0.4,0.4,0.4,0.4,0.4]) #,fixed_LocErr = 0.02, fixed_D = 0.1)
    
    q = [param + ' = ' + str(np.round(model_fit.params[param].value, 4)) for param in model_fit.params]
    print(q)
    
    fit_params.append(model_fit.params)
    
    F0s.append(model_fit.params['F0'].value)
    p01s.append(model_fit.params['p01'].value)
    p10s.append(model_fit.params['p10'].value)
    D0s.append(model_fit.params['D0'].value)
    D1s.append(model_fit.params['D1'].value)
    LocErrs.append(model_fit.params['LocErr'].value)



model_fit = get_2DSPT_params(data_list, dt = n*dt,states_nb =2, nb_substeps = 1, verbose = 1, frame_len = 5, steady_state = False) #,fixed_LocErr = 0.02, fixed_D = 0.1)


preds = predict_Bs(data_list, dt, model_fit.params, states_nb=3, frame_len=7)

len(preds)

idx = 1



for track, Bs in zip(data_list[idx], preds[idx]):
    plt.plot(track[:,0], track[:,1], ':', c = [0.5,0.5,0.5])
    plt.scatter(track[:,0], track[:,1], c = Bs)


3**7

vals_array = []
for param in ['LocErr', 'D0', 'D1', 'D2', 'F0', 'F1', 'p01','p02','p10','p12','p20','p21']:
    vals = []
    for k in range(len(fit_params)):
        vals.append(fit_params[k][param].value)
    vals_array.append(vals)
    
vals_array = np.array(vals_array)


'''
results with steady state

array([[2.10345519e-02, 1.88391569e-02, 1.91997038e-02],
       [1.00000000e-12, 1.00000000e-12, 1.00000000e-12],
       [3.86934179e-02, 1.41511476e-02, 2.44912509e-02],
       [8.28679555e-02, 7.92984068e-02, 8.18700000e-02],
       [1.07849344e-01, 1.31717784e-01, 9.38062908e-02],
       [7.44890046e-02, 6.37078227e-02, 6.85967767e-02],
       [6.05076481e-02, 3.72522964e-02, 2.94409418e-02],
       [1.06956448e-01, 9.88032593e-02, 1.06886517e-01],
       [1.05253967e-01, 3.30440577e-02, 2.62824537e-02],
       [1.28150843e-02, 1.71788211e-01, 8.07437432e-02],
       [1.24998275e-02, 1.96573138e-02, 1.31154750e-02],
       [2.77515272e-03, 1.01204164e-02, 5.46791231e-03]])
'''
k=0

ex = '*/*.xml'
movies = glob('/run/user/1000/gvfs/smb-share:server=gaia.pasteur.fr,share=qbact/Andrey/data/2019/0823-29-D-cyc-treatment-recovery-AV51/0828-AV51-5-OD0.16-steady-state-10h00/'+ex )
dts = [0.06]*20

lengths = np.arange(5,21)
#for min_len in [5,7,9]:#[5,7,9,13]: #[3,4,5,7,8,10]
    #lengths = [min_len]
data = {}
for l in lengths:
    data[str(l)] = np.array([[[]]*l]*2).T

for movie in movies[k:k+1]:
    cur_data, frames = read_multi_trackmate_xml(movie, lengths, dist_th = 0.3)
    for l in lengths:
        if len(cur_data[str(l)])>0:
            data[str(l)] = np.concatenate((data[str(l)], cur_data[str(l)]))
            
all_len = []
for l in lengths:
    all_len.append(len(data[str(l)]))
print(all_len)
k+=1
all_lens.append(all_len)

nb_substeps = 1
data_list = []
for l in lengths:
    if len(data[str(l)])>0:
        data_list.append(np.array(data[str(l)][:,:])[:,np.arange(0,l,n)])

model_fit = get_2DSPT_params(data_list, dt = n*dt,states_nb =3, nb_substeps = 1, verbose = 1, frame_len = 4, steady_state = False, vary_params = [True, False, True, True, True, True,True, True, True, True,True, True],estimated_vals = [0.029, 1e-12, 0.07, 0.09, 0.3, 0.3, 0.1,0.1,0.1,0.1,0.1,0.1],min_values = [0.007, 1e-12, 0.0001, 0.001, 0.001,0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], max_values = [0.6,1,1,1,1,1,0.4,0.4,0.4,0.4,0.4,0.4]) #,fixed_LocErr = 0.02, fixed_D = 0.1)

q = [param + ' = ' + str(np.round(model_fit.params[param].value, 4)) for param in model_fit.params]
print(q)

model_fit = get_2DSPT_params(data_list, dt = n*dt,states_nb =3, nb_substeps = 1, verbose = 1, frame_len = 4, steady_state = False, vary_params = [False, False, False, False, False, False,False, False, False, False,False, False],estimated_vals = [0.029, 1e-12, 0.02, 0.08, 0.33, 0.33, 0.1,0.1,0.1,0.1,0.1,0.1],min_values = [0.007, 1e-12, 0.0001, 0.001, 0.001,0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], max_values = [0.6,1,1,1,1,1,0.4,0.4,0.4,0.4,0.4,0.4]) #,fixed_LocErr = 0.02, fixed_D = 0.1)

model_fit = get_2DSPT_params(data_list, dt = n*dt,states_nb =2, nb_substeps = 1, verbose = 1, frame_len = 4, steady_state = False, vary_params = [False, False, False, False,False, False],estimated_vals = [0.02, 1e-12, 0.07, 0.5,0.1,0.1]) #,fixed_LocErr = 0.02, fixed_D = 0.1)

preds = predict_Bs(data_list, dt, model_fit.params, states_nb=3, frame_len=8)

plt.figure()
for idx in range(len(lengths)):
    for track, Bs in zip(data_list[idx], preds[idx]):
        plt.plot(track[:,0], track[:,1], ':', c = [0.5,0.5,0.5])
        #Bs = np.concatenate((Bs, np.zeros((len(Bs),1))),1)
        plt.scatter(track[:,0], track[:,1], c = Bs)

plt.figure()
idx = -1
for track, Bs in zip(data_list[idx], preds[idx]):
    plt.plot(track[:,0], track[:,1], ':', c = [0.5,0.5,0.5])
    #Bs = np.concatenate((Bs, np.zeros((len(Bs),1))),1)
    plt.scatter(track[:,0], track[:,1], c = Bs)


np.concatenate((Bs, np.zeros((len(Bs),1))),1)








