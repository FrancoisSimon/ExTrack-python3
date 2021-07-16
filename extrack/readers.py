import xmltodict
from glob import glob
import numpy as np

def read_trackmate_xml(path, lengths=[6,7,8,9,10,11,12,13,14,15], dist_th = 0.3, start_frame = 0):
    """
    Converts xml output from trackmate to a list of arrays of tracks
    each element of the list is an array composed of several tracks (dim 0), of a fix number of position (dim 1)
    with x and y coordinates (dim 2)
    path : path to xml file
    lengths : lengths used for the arrays, list of intergers, tracks with n localization will be added 
    to the array of length n if n is in the list, if n > max(lenghs)=k the k first steps of the track will be added.
    dist_th : maximum distance allowed to connect consecutive peaks
    start_frame : first frame considered 
    """
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
    nb_peaks = 0
    for l in lengths:
        traces[str(l)] = []
        frames[str(l)] = []
    try:
        for i, particle in enumerate(data['Tracks']['particle']):
            track = [(float(d['@x']), float(d['@y']), float(d['@t'])*framerate, int(d['@t']), i) for d in particle['detection']]
            #print(len(track))
            track = np.array(track)
            nb_peaks += len(track)
            no_zero_disp = np.min((track[1:,0] - track[:-1,0])**2) * np.min((track[1:,1] - track[:-1,1])**2)
            no_zero_disp = no_zero_disp > 0
            dists = np.sum((track[1:, :2] - track[:-1, :2])**2, axis = 1)**0.5
            if no_zero_disp and track[0, 3] > start_frame and track[0, 3] < 10000 and np.all(dists<dist_th):
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
    '''
    traces_list = []
    frames_list = []
    for l in lengths:
        if len(traces[str(l)])>0:
            traces_list.append(np.array(traces[str(l)]))
            frames_list.append(np.array(frames[str(l)]))
    '''
    return traces, frames, nb_peaks
  
def read_matlab_table():
  
  
def read_CSV_table():
  
  
