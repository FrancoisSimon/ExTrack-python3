from extrack.simulate_tracks import get_tracks

track_lengths = [7,8,9,10,11] # create arrays of tracks of specified number of localizations  
track_nb_dist = [1000, 800, 700, 600, 550] # list of number of tracks per array
LocErr = 0.02 # Localization error in um
Ds = [0, 0.1] # diffusion coef of each state in um^2/s
TrMat = np.array([[0.9,0.1], [0.2,0.8]]) # transition matrix e.g. np.array([[1-p01,p01], [p10,1-p10]])
initial_fractions = None # fraction in each states at the beginning of the tracks, if None assums steady state determined by TrMat
dt = 0.02 # time in between positions in s
nb_dims = 2 # number of spatial dimensions : 2 for (x,y) and 3 for (x,y,z)

all_Css, all_Bss = get_tracks(track_lengths = track_lengths,
                              track_nb_dist = track_nb_dist,
                              LocErr = LocErr,
                              Ds = Ds,
                              TrMat = TrMat,
                              initial_fractions = initial_fractions, 
                              dt = dt,
                              nb_dims = nb_dims)

model_fit, preds = auto_fitting_2states(all_Css,dt)

model_fit.params
