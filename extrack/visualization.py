import numpy as np
from matplotlib import pyplot as plt
from extrack.histograms import len_hist
from matplotlib import cm

def visualize_states_durations(all_Css,
                               params,
                               dt,
                               cell_dims = [1,None,None],
                               states_nb = 2,
                               max_nb_states = 500,
                               long_tracks = True,
                               min_len = 20,
                               steps = False):
    
    len_hists = len_hist(all_Css, params, dt, cell_dims=cell_dims, states_nb=states_nb, nb_substeps=1, max_nb_states = max_nb_states)
    
    if step:
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
    plt.xlim([0,min_len*dt])
    plt.ylim([0.001,0.5])
    plt.xlabel('state duration (%s)'(step_type))
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
                lim = 0.4
                ):
    
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

    nb_plots = 5
    lim = 0.2
    for k in range(nb_plots**2):
        plt.subplot(nb_plots, nb_plots, k+1)
        #plt.imshow(img)
        offset = 0*5**2
        track = binding_tracks[k+offset]
        pred = binding_preds[k+offset]
        
        plt.plot(track[:,0], track[:,1], 'k:', alpha = 0.2)
        plt.scatter(track[:,0], track[:,1], c = cm.brg(pred[:,1]*0.5), s=5)
        plt.scatter(track[0,0], track[0,1], marker = 'x', c='k', s=5, alpha = 0.5)
        plt.xlim([np.mean(track[:,0][track[:,0]>-np.Inf]) - lim, np.mean(track[:,0][track[:,0]>-np.Inf]) + lim])
        plt.ylim([np.mean(track[:,1][track[:,0]>-np.Inf]) - lim, np.mean(track[:,1][track[:,0]>-np.Inf]) + lim])
