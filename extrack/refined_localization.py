import numpy as np

GPU_computing = 0

if GPU_computing :
    import cupy as cp
    from cupy import asnumpy
else:
    # if CPU computing :
    import numpy as cp
    def asnumpy(x):
        return x

from matplotlib import pyplot as plt
from extrack.extrack import gaussian, log_integrale_dif, first_log_integrale_dif, prod_2GaussPDF, prod_3GaussPDF, ds_froms_states, fuse_tracks, get_all_Bs, get_Ts_from_Bs
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

def get_LC_Km_Ks(Cs, LocErr, ds, Fs, TR_params, nb_substeps=1, frame_len = 4):
    '''
    variation of the main function to extract LC, Km and Ks for all positions
    '''
    nb_Tracks = len(Cs)
    nb_locs = len(Cs[0]) # number of localization per track
    nb_dims = len(Cs[0,0]) # number of spatial dimentions (x, y) or (x, y, z)
    Cs = Cs.reshape((nb_Tracks,1,nb_locs, nb_dims))
    Cs = cp.array(Cs)
    
    nb_states = TR_params[0]
    all_Km = []
    all_Ks = []
    all_LP = [] 
    
    preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    
    TrMat = cp.array(TR_params[1]) # transition matrix of the markovian process
    current_step = 1
    
    cur_Bs = get_all_Bs(nb_substeps + 1, nb_states) # get initial sequences of states
    cur_Bs = cur_Bs[None,:,:] # include dim for different tracks
    
    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
    
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds)
    Fs = cp.array(Fs)
    
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
    
    LP = LT # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    
    cur_ds = ds_froms_states(ds, cur_states)
    
    # inject the first position to get the associated Km and Ks :
    Km, Ks = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr, cur_ds)
    
    Ks = cp.repeat(Ks, nb_Tracks, axis = 0)
    
    current_step += 1
    Km = cp.repeat(Km, cur_nb_Bs, axis = 1)
    removed_steps = 0
    
    all_Km.append(Km)
    all_Ks.append(Ks)
    all_LP.append(LP)
    
    while current_step <= nb_locs-1:
        # update cur_Bs to describe the states at the next step :
        cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step

        cur_ds = ds_froms_states(ds, cur_states)
        
        LT = get_Ts_from_Bs(cur_states, TrMat)
        
        # repeat the previous matrix to account for the states variations due to the new position
        Km = cp.repeat(Km, nb_states**nb_substeps , axis = 1)
        Ks = cp.repeat(Ks, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        # inject the next position to get the associated Km, Ks and Constant describing the integral of 3 normal laws :
        Km, Ks, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr, cur_ds, Km, Ks)
        #print('integral',time.time() - t0); t0 = time.time()
        LP += LT + LC # current (log) constants associated with each track and sequences of states
        del LT, LC
        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states

        ''''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the Km and Ks of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        if current_step < nb_locs-1:
            while cur_nb_Bs >= nb_states**frame_len:
                newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
                log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,nb_locs-current_step] - Km)**2,axis=2)/(2*newKs**2)
                LF = 0 #cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
                
                test_LP = LP + log_integrated_term + LF
                
                if np.max(test_LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
                    test_LP = test_LP - (np.max(test_LP)-600)

                P = np.exp(test_LP)
                for state in range(nb_states):
                    B_is_state = cur_Bs[:,:,-1] == state
                    preds[:,nb_locs-current_step+frame_len-2, state] = asnumpy(np.sum(B_is_state*P,axis = 1)/np.sum(P,axis = 1))
                cur_Bs.shape
                cur_Bs = cur_Bs[:,:cur_nb_Bs//nb_states, :-1]
                Km, Ks, LP = fuse_tracks(Km, Ks, LP, cur_nb_Bs, nb_states)
                cur_nb_Bs = len(cur_Bs[0])
                removed_steps += 1
        
        all_Km.append(Km)
        all_Ks.append(Ks)
        all_LP.append(LP)
        
        current_step += 1

    newKs = cp.array((Ks**2 + LocErr**2)**0.5)[:,:,0]
    log_integrated_term = -cp.log(2*np.pi*newKs**2) - cp.sum((Cs[:,:,0] - Km)**2,axis=2)/(2*newKs**2)
    LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LF
    
    pred_LP = LP
    if np.max(LP)>600: # avoid overflow of exponentials, mechanically also reduces the weight of longest tracks
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    for state in range(nb_states):
        B_is_state = cur_Bs[:,:] == state
        preds[:,0:frame_len, state] = asnumpy(np.sum(B_is_state*P[:,:,None],axis = 1)/np.sum(P[:,:,None],axis = 1))
    return LP, cur_Bs, preds, all_Km, all_Ks, all_LP

def get_pos_PDF(Cs, LocErr, ds, Fs, TR_params, frame_len = 7):
    ds = np.array(ds)
    # get Km, Ks and LC forward
    LP1, final_Bs1, preds1, all_Km1, all_Ks1, all_LP1 = get_LC_Km_Ks(Cs, LocErr, ds, Fs, TR_params, nb_substeps=1, frame_len = frame_len)
    #get Km, Ks and LC backward
    TR_params2 = [TR_params[0], TR_params[1].T] # transpose the matrix for the backward transitions
    Cs2 = Cs[:,::-1,:] # inverse the time steps
    LP2, final_Bs2, preds2, all_Km2, all_Ks2, all_LP2 = get_LC_Km_Ks(Cs2, LocErr, ds, np.ones(TR_params2[0],)/TR_params2[0], TR_params2, nb_substeps=1, frame_len = frame_len) # we set a neutral Fs so it doesn't get counted twice

    # do the approximation for the first position, product of 2 gaussian PDF, (integrated term and localization error)    
    sig, mu, LC = prod_2GaussPDF(LocErr,all_Ks1[-1], Cs[:,None,0], all_Km1[-1])
        
    LP = all_LP1[-1] + LC
    all_pos_means = [mu]
    all_pos_stds = [sig]
    all_pos_weights = [LP]
    all_pos_Bs = [final_Bs1]
    
    for k in range(1,Cs.shape[1]-1):
        '''
        we take the corresponding Km1, Ks1, LP1, Km2, Ks2, LP2
        which are the corresponding stds and means of the resulting 
        PDF surrounding the position k
        with localization uncertainty, we have 3 gaussians to compress to 1 gaussian * K
        This has to be done for all combinations of set of consective states before and after.step k
        to do so we set dim 1 as dim for consecutive states computed by the forward proba and 
        dim 2 for sets of states computed by the backward proba.
        
        '''
        LP1 = all_LP1[-1-k][:,:,None]
        Km1 = all_Km1[-1-k][:,:,None]
        Ks1 = all_Ks1[-1-k][:,:,None]
        LP2 = all_LP2[k-1][:,None]
        Km2 = all_Km2[k-1][:,None]
        Ks2 = all_Ks2[k-1][:,None]
        
        nb_Bs1 = Ks1.shape[1]
        nb_Bs2 = Ks2.shape[2]
        nb_tracks = Km1.shape[0]
        nb_dims = Km1.shape[3]
        nb_states = TR_params[0]
        LP2.shape
        Bs2_len = np.min([k+1, frame_len-1])
        cur_Bs2 = get_all_Bs(Bs2_len, nb_states)
        # we must reorder the metrics so the Bs from the backward terms correspond to the forward terms
        indexes = np.sum(cur_Bs2 * nb_states**np.arange(Bs2_len)[::-1][None],1).astype(int)
        indexes.shape
        LP2 = LP2[:,:,indexes]
        Km2 = Km2[:,:,indexes]
        Ks2 = Ks2[:,:,indexes]
        Km2.shape
        # we must associate only forward and backward metrics that share the same state at position k as followed :
        slice_len = nb_Bs1//nb_states
        new_LP1 = LP1[:,0:slice_len:nb_states]
        new_Km1 = Km1[:,0:slice_len:nb_states]
        new_Ks1 = Ks1[:,0:slice_len:nb_states]
        for i in range(1,nb_states):
            new_LP1 = np.concatenate((new_LP1,LP1[:,i*slice_len+i:(i+1)*slice_len:nb_states]), 1)
            new_Km1 = np.concatenate((new_Km1,Km1[:,i*slice_len+i:(i+1)*slice_len:nb_states]), 1)
            new_Ks1 = np.concatenate((new_Ks1,Ks1[:,i*slice_len+i:(i+1)*slice_len:nb_states]), 1)
        
        LP1 = new_LP1
        Km1 = new_Km1
        Ks1 = new_Ks1
        
        cur_nb_pos = np.round((np.log(nb_Bs2)+np.log(nb_Bs1//nb_states))/np.log(nb_states)).astype(int)
        cur_Bs = get_all_Bs(cur_nb_pos, nb_states)[None]

        sig, mu, LC = prod_3GaussPDF(Ks1,LocErr,Ks2, Km1, Cs[:,None,None,k], Km2)
        LP = LP1 + LP2 + LC
        sig.shape
        sig = sig.reshape((nb_tracks,(nb_Bs1//nb_states)*nb_Bs2,1))
        mu = mu.reshape((nb_tracks,(nb_Bs1//nb_states)*nb_Bs2, nb_dims))
        LP = LP.reshape((nb_tracks,(nb_Bs1//nb_states)*nb_Bs2))
        
        all_pos_means.append(mu)
        all_pos_stds.append(sig)
        all_pos_weights.append(LP)
        all_pos_Bs.append(cur_Bs)
        
    sig, mu, LC = prod_2GaussPDF(LocErr,all_Ks2[-1], Cs[:,None,-1], all_Km2[-1])
    LP = all_LP2[-1] + LC
    
    cur_Bs2 = get_all_Bs(Bs2_len, nb_states)
    all_pos_means.append(mu)
    all_pos_stds.append(sig)
    all_pos_weights.append(LP)
    all_pos_Bs.append(final_Bs2)

    return all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs

def get_best_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds):
    nb_Bs = []
    nb_pos = len(all_pos_weights)
    for weights in all_pos_weights:
        nb_Bs.append(weights.shape[1])
    nb_Bs = np.max(nb_Bs)
    nb_states = (np.max(all_pos_Bs[0])+1).astype(int)
    max_frame_len = (np.log(nb_Bs) // np.log(nb_states)).astype(int)
    mid_frame_pos = ((max_frame_len-0.1)//2).astype(int) # -0.1 is used for mid_frame_pos to be the good index for both odd and pair numbers
    mid_frame_pos = np.max([1,mid_frame_pos])
    best_Bs = []
    best_mus = []
    best_sigs = []
    for k, (weights, Bs, mus, sigs)  in enumerate(zip(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)):
        if k <= nb_pos/2 :
            idx = np.min([k, mid_frame_pos])
        else :
            idx = np.max([-mid_frame_pos-1, k - nb_pos])
        best_args = np.argmax(weights, 1)
        best_Bs.append(Bs[0,best_args][:,idx])
        best_sigs.append(sigs[[np.arange(len(mus)), best_args]])
        best_mus.append(mus[[np.arange(len(mus)), best_args]])
    best_Bs = np.array(best_Bs).T.astype(int)
    best_sigs = np.transpose(np.array(best_sigs), (1,0,2))
    best_mus = np.transpose(np.array(best_mus), (1,0,2))
    return best_mus, best_sigs, best_Bs

def save_gifs(Cs, all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs, gif_pathnames = './tracks', lim = None, nb_pix = 200, fps=1):
    best_mus, best_sigs, best_Bs = get_best_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)
    for ID in range(len(Cs)):
        all_images = []
        Cs_offset = np.mean(Cs[ID], 0)
        Cs[ID] = Cs[ID] 
        
        if lim == None:
            cur_lim = np.max(np.abs(Cs[ID]))*1.1
        else:
            cur_lim = lim
        pix_size = nb_pix / (2*cur_lim)
        for k in range(len(all_pos_means)):
                
            sig = all_pos_stds[k]
            mu = all_pos_means[k]
            LP = all_pos_weights[k]
            mu.shape
            fig = plt.figure()            
            plt.plot((Cs[ID, :,1] - Cs_offset[1] + cur_lim)*pix_size-0.5, (Cs[ID, :,0]- Cs_offset[0]+cur_lim)*pix_size-0.5)
            plt.scatter((best_mus[ID, :,1] - Cs_offset[1] + cur_lim)*pix_size-0.5, (best_mus[ID, :,0] - Cs_offset[0]+cur_lim)*pix_size-0.5, c='r', s=3)
            best_mus.shape

            P_xs = gaussian(np.linspace(-cur_lim,cur_lim,nb_pix)[None,:,None], sig[ID][:,:,None], mu[ID][:,:1,None] - Cs_offset[0]) * np.exp(LP[ID]-np.max(LP[ID]))[:,None]
            P_ys = gaussian(np.linspace(-cur_lim,cur_lim,nb_pix)[None,:,None], sig[ID][:,:,None], mu[ID][:,1:,None] - Cs_offset[1]) * np.exp(LP[ID]-np.max(LP[ID]))[:,None]
            
            heatmap = np.sum(P_xs[:,:,None]*P_ys[:,None] * np.exp(LP[ID]-np.max(LP[ID]))[:,None,None],0)
            
            heatmap = heatmap/np.max(heatmap)
            plt.imshow(heatmap)
            plt.xticks(np.linspace(0,nb_pix-1, 5), np.round(np.linspace(-cur_lim,cur_lim, 5), 2))
            plt.yticks(np.linspace(0,nb_pix-1, 5), np.round(np.linspace(-cur_lim,cur_lim, 5), 2))
            canvas = FigureCanvas(fig)
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()
            image = np.fromstring(s, dtype='uint8').reshape((height, width, 4))
            
            all_images.append(image)
            plt.close()
        
        imageio.mimsave(gif_pathnames + str(ID)+'.gif', all_images,fps=fps)

        
               
def do_gifs_from_params(all_Cs, params, dt, gif_pathnames = './tracks', frame_len = 9, states_nb = 2, nb_pix = 200, fps = 1):
    for Cs in all_Cs:
        LocErr, ds, Fs, TR_params = extract_params(params, dt, states_nb, nb_substeps = 1)
        all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs = get_pos_PDF(Cs, LocErr, ds, Fs, TR_params, frame_len = frame_len)
        save_gifs(Cs, all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs, gif_pathnames = gif_pathnames + '_' + str(len(Cs[0])) + '_pos', lim = None, nb_pix = nb_pix, fps=fps)    

def get_refined_pos_from_params(all_Cs, params, dt, gif_pathnames = './tracks', frame_len = 9, states_nb = 2):
    all_best_mus = []
    all_best_sigs = []
    all_best_Bs = []
    for Cs in all_Cs:
        LocErr, ds, Fs, TR_params = extract_params(params, dt, states_nb, nb_substeps = 1)
        all_pos_means, all_pos_stds, all_pos_weights, all_pos_Bs = get_pos_PDF(Cs, LocErr, ds, Fs, TR_params, frame_len = frame_len)
        best_mus, best_sigs, best_Bs = get_best_estimates(all_pos_weights, all_pos_Bs, all_pos_means, all_pos_stds)
        all_best_mus.append(best_mus)
        all_best_sigs.append(best_sigs)
        all_best_Bs.append(best_Bs)
    return all_best_mus, all_best_sigs, all_best_Bs
    
   
