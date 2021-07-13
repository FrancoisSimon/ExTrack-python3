from matplotlib import pyplot as plt


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
