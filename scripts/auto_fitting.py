from Tracking import get_2DSPT_params
from Tracking import predict_Bs

def auto_fitting_2states(all_Cs,dt, steady_state = True, estimated_vals = {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05}, vary_params = {'LocErr' : True, 'D0' : False, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True}):
                estimated_vals = {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05}
    
    model_fit = get_2DSPT_params(all_Cs, dt, nb_substeps = 1, states_nb = 2, do_frame = 1,frame_len = 4, verbose = 0, vary_params = [vary_params, estimated_vals = estimated_vals, steady_state = steady_state)
    
    estimated_vals = [model_fit.params['LocErr'].value, model_fit.params['D0'].value, model_fit.params['D1'].value, model_fit.params['F0'].value, model_fit.params['p01'].value, model_fit.params['p10'].value]
    tr_freq = estimated_vals[3]*estimated_vals[4] + (1-estimated_vals[3])*estimated_vals[5]
    DLR = (2*dt*estimated_vals[2])**0.5/estimated_vals[0]
    
    frame_lens = [6,6,5]
    nbs_substeps = [1,2,3]
    nb_substeps = 1

    if DLR < 1.5:
        frame_len = 8
        nb_substeps = 1
    elif DLR < 5:
        if tr_freq > 0.15:
            nb_substeps = 2
        if  tr_freq < 0.15:
            nb_substeps = 1
        frame_len = frame_lens[nb_substeps-1]
    else :
        frame_lens = [6,6,5]
        if  tr_freq < 0.15:
            nb_substeps = 1
        if tr_freq > 0.15:
            nb_substeps = 2
        if tr_freq > 0.3:
            nb_substeps = 3
        frame_len = frame_lens[nb_substeps-1]

    keep_running = 1
    res_val = 0
    for kk in range(10):
        if keep_running:
            model_fit = get_2DSPT_params(all_Css, dt, nb_substeps = nb_substeps, states_nb = 2, do_frame = 1,frame_len = frame_len, verbose = 0, vary_params = [True, False, True, True, True, True], estimated_vals = [model_fit.params['LocErr'], model_fit.params['D0'], model_fit.params['D1'], model_fit.params['F0'], model_fit.params['p01'], model_fit.params['p10']], steady_state = True)
            if res_val - 0.1 > model_fit.residual:
                res_val = model_fit.residual
            else:
                keep_running = 0

        q = [param + ' = ' + str(np.round(model_fit.params[param].value, 4)) for param in model_fit.params]
        print(model_fit.residual[0], q)
        
    preds = predict_Bs(all_Cs, dt, model_fit.params, states_nb, frame_len = 12)
    return model_fit, preds
  
def auto_fitting_3states(all_Cs,dt, steady_state = True, vary_params = { 'LocErr' : True, 'D0' : False, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True},
                                                         estimated_vals = { 'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1},
                                                         min_values = { 'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001},
                                                         max_values = { 'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1}):
    '''
    all_Cs is a list of numpy arrays of tracks, each numpy array have the following dimensions : dim 0 = track ID, dim 1 = time position, dim 2 = x and y positions
    dt is a scalar : the time per frame
    steady_state = True assume fractions are determined by the rates, fractions and rates are independent if False 
    the other optional lists allow to fit or fix parameter fitting (vary_params), chose values (estimated_vals), 
    and limits for the fit (min_values and max_values)
    '''
    model_fit = get_2DSPT_params(all_Cs, dt, nb_substeps = 1, states_nb = 2, do_frame = 1,frame_len = 3, verbose = 0, steady_state = steady_state, vary_params = vary_params, estimated_vals = estimated_vals)
    
    keep_running = 1
    res_val = 0
    for kk in range(10):
        if keep_running:
            model_fit = get_2DSPT_params(all_Css, dt, nb_substeps = 1, states_nb = 2, do_frame = 1,frame_len = frame_len, verbose = 0, vary_params = vary_params, estimated_vals = [model_fit.params['LocErr'], model_fit.params['D0'], model_fit.params['D1'], model_fit.params['F0'], model_fit.params['p01'], model_fit.params['p10']], steady_state = True)
            if res_val - 0.1 > model_fit.residual:
                res_val = model_fit.residual
            else:
                keep_running = 0
    
        q = [param + ' = ' + str(np.round(model_fit.params[param].value, 4)) for param in model_fit.params]
        print(model_fit.residual[0], q)
        
    preds = predict_Bs(all_Cs, dt, model_fit.params, states_nb, frame_len = 12)
    return model_fit, preds
  
