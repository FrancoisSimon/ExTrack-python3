
#def dict_pred_to_df_pred(all_Cs, all_Bs):

def save_params(params, path = '.', fmt = 'json', file_name = 'params'):
    save_params = {}
    [save_params.update({param : params[param].value}) for param in params]
    
    '''
    available formats : json, npy, csv
    ''''        
    if fmt == 'npy':
        np.save(path + '/' + file_name, save_params)
    elif fmt == 'pkl':
        with open(path + '/' + file_name + ".pkl", "wb") as tf:
            pickle.dump(save_params,tf)
    elif fmt == 'json':
        with open(path + '/' + file_name + ".json", "w") as tf:
            json.dump(save_params,tf)
    elif fmt == 'csv':
        with open(path + '/' + file_name + ".csv", 'w') as tf:
            for key in save_params.keys():
                tf.write("%s,%s\n"%(key,save_params[key]))
    else :
        raise ValueError("format not supported, use one of : 'json', 'pkl', 'npy', 'csv'")
        
def pred_2_matrix(all_Css, pred_Bss, dt, all_frames = None):
    row_ID = 0
    nb_pos = 0
    for len_ID in all_Css:
       all_Cs = all_Css[len_ID]
       nb_pos += all_Cs.shape[0]*all_Cs.shape[1]
    
    matrix = np.empty((nb_pos, 4+pred_Bss[list(pred_Bss.keys())[0]].shape[2]))
    #TRACK_ID,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,PRED_0, PRED_1,(PRED_2 etc)
    track_ID = 0
    for len_ID in all_Css:
        all_Cs = all_Css[len_ID]
        pred_Bs = pred_Bss[len_ID]
        if all_frames != None:
            all_frame = all_frames[len_ID]
        else:
            all_frame = np.arange(all_Cs.shape[0]*all_Cs.shape[1]).reshape((all_Cs.shape[0],all_Cs.shape[1])) 
        for track, preds, frames in zip(all_Cs, pred_Bs, all_frame):
            track_IDs = np.full(len(track),track_ID)[:,None]
            frames = frames[:,None]
            cur_track = np.concatenate((track, track_IDs,frames,preds ),1)
            
            matrix[row_ID:row_ID+cur_track.shape[0]] = cur_track
            row_ID += cur_track.shape[0]
        track_ID+=1
    return matrix

def save_pred_2_CSV(all_Css, pred_Bss, dt, all_frames = None):
    track_ID = 0
    
    preds_header_str_fmt = ''
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_header_str_fmt = preds_header_str_fmt + 'PRED_%s,'%(k)
        preds_str_fmt = preds_str_fmt + '%s,'
        
    with open(path, 'w') as f:
        f.write('TRACK_ID,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,%s\n'%(preds_header_str_fmt))
    
        for len_ID in all_Css:
            all_Cs = all_Css[len_ID]
            pred_Bs = pred_Bss[len_ID]
            if all_frames != None:
                all_frame = all_frames[len_ID]
            else:
                all_frame = np.arange(all_Cs.shape[0]*all_Cs.shape[1]).reshape((all_Cs.shape[0],all_Cs.shape[1])) 
            for track, preds, frames in zip(all_Cs, pred_Bs, all_frame):
                track_ID+=1
                for pos, p, frame in zip(track, preds, frames):
                    preds_str = preds_str_fmt%(tuple(p))
                    f.write('%s,%s,%s,%s,%s,%s,%s\n'%(track_ID,p[0],p[1],0.0,dt* frame*1000, frame, preds_str))

def save_pred_2_xml(all_Css, pred_Bss, path, dt, all_frames = None):
    track_ID = 0
    for len_ID in all_Css:
       all_Cs = all_Css[len_ID]
       track_ID += len(all_Cs)
    
    preds_str_fmt = ''
    for k in range(pred_Bss[list(pred_Bss.keys())[0]].shape[2]):
        preds_str_fmt = preds_str_fmt + 'pred_%s="%s" '%(k,'%s')
    
    track_ID = 0
    with open(path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<Tracks nTracks="%s" spaceUnits="µm" frameInterval="%s" timeUnits="ms">\n'%(track_ID, dt))
    
        for len_ID in all_Css:
            all_Cs = all_Css[len_ID]
            pred_Bs = pred_Bss[len_ID]
            if all_frames != None:
                all_frame = all_frames[len_ID]
            else:
                all_frame = np.arange(all_Cs.shape[0]*all_Cs.shape[1]).reshape((all_Cs.shape[0],all_Cs.shape[1])) 
            for track, preds, frames in zip(all_Cs, pred_Bs, all_frame):
                track_ID+=1
                f.write('  <particle nSpots="%s">\n'%(len_ID))
                for pos, p, frame in zip(track, preds, frames):
                    preds_str = preds_str_fmt%(tuple(p))
                    f.write('    <detection t="%s" x="%s" y="%s" z="%s" %s/>\n'%(frame,p[0],p[1],0.0, preds_str))


