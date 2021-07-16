

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
    
        
    

