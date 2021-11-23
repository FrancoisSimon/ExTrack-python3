import numpy as np
from matplotlib import pyplot as plt
from extrack.histograms import len_hist
from matplotlib import cm

def visualize_states_durations(all_tracks,
                               params,
                               dt,
                               cell_dims = [1,None,None],
                               nb_states = 2,
                               max_nb_states = 500,
                               long_tracks = True,
                               nb_steps_lim = 20,
                               steps = False):
    
    len_hists = len_hist(all_tracks, params, dt, cell_dims=cell_dims, nb_states=nb_states, nb_substeps=1, max_nb_states = max_nb_states)
        
    if steps:
        step_type = 'step'
        dt = 1
    else:
        step_type = 's'
    
    plt.figure(figsize = (3,3))
    for k, hist in enumerate(len_hists.T):
        plt.plot(np.arange(1,len(hist)+1)*dt, hist/np.sum(hist), label='state %s'%k)
    
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.xlim([0,nb_steps_lim*dt])
    plt.ylim([0.001,0.5])
    plt.xlabel('state duration (%s)'%(step_type))
    plt.ylabel('fraction')
    plt.tight_layout()

def visualize_tracks(DATA,
                     track_length_range = [10,np.inf]):
    
    nb_states = 0
    for param in list(DATA.keys()):
        if param.find('pred')+1:
            nb_states += 1
    
    plt.figure()
    DATA['X']
    for ID in np.unique(DATA['track_ID'])[::-1]:
        print(ID)
        track = DATA[DATA['track_ID'] ==ID ]
        if track_length_range[0] < len(track) > track_length_range[0]:
            if nb_states == 2 :
                pred = track['pred_1']
                pred = cm.brg(pred*0.5)
            else:
                pred = track[['pred_2', 'pred_1', 'pred_0']].values
            
            plt.plot(track['X'], track['Y'], 'k:', alpha = 0.2)
            plt.scatter(track['X'], track['Y'], c = pred, s=5)
            #plt.scatter(track['X'], track['X'], marker = 'x', c='k', s=5, alpha = 0.5)

def plot_tracks(DATA,
                max_track_length = 50,
                fig_dims = [5,5],
                lim = 0.4 ):
    
    nb_states = 0
    for param in list(DATA.keys()):
        if param.find('pred')+1:
            nb_states += 1
    
    plt.figure(figsize=(10,10))
    
    for ID in np.unique(DATA['track_ID'])[::-1]:
        track = DATA[DATA['track_ID'] ==ID]
        if len(track) > max_track_length:
            DATA.drop((DATA[DATA['track_ID'] == ID]).index, inplace=True)
    
    for k, ID in enumerate(np.unique(DATA['track_ID'])[::-1][:np.product(fig_dims)]):
        plt.subplot(fig_dims[0], fig_dims[1], k+1)
        print(ID)
        track = DATA[DATA['track_ID'] ==ID ]
        if nb_states == 2 :
            pred = track['pred_1']
            pred = cm.brg(pred*0.5)
        else:
            pred = track[['pred_2', 'pred_1', 'pred_0']].values
        
        plt.plot(track['X'], track['Y'], 'k:', alpha = 0.2)
        plt.scatter(track['X'], track['Y'], c = pred, s=5)
        plt.xlim([np.mean(track['X']) - lim, np.mean(track['X']) + lim])
        plt.ylim([np.mean(track['Y']) - lim, np.mean(track['Y']) + lim])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.yticks(fontsize = 6)
        plt.xticks(fontsize = 6)
    plt.tight_layout(h_pad = 1, w_pad = 1)
