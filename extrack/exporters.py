

def pred_2_csv():
    return

def pred_to_xml():
    return


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

