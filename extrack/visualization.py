from matplotlib import pyplot as plt

def visualize_states_durations(all_Css,
                               params,
                               dt,
                               cell_dims = cell_dims,
                               states_nb = states_nb,
                               max_nb_states = 500,
                               long_tracks = True,
                               min_len = 20):
    len_hists = len_hist(all_Css, params, dt, cell_dims=cell_dims, states_nb=states_nb, nb_substeps=1, max_nb_states = max_nb_states)
    
    plt.figure(figsize = (3,3))
    for k, hist in enumerate(len_hists.T):
        plt.plot(np.arange(1,len(hist)+1), hist/np.sum(hist), label='state %s'%k)

    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.xlim([0,20])
    plt.ylim([0.001,0.5])
    plt.xlabel('state duration (in step)')
    plt.ylabel('fraction')
    plt.tight_layout()

plt.figure()
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
