# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

# TensorFlow options.
tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.
tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']           = True      # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.
#----------------------------------------------------------------------------
desc        = 'prog'                                        # Description string included in result subdir name.
random_seed = 8001                                          # Global random seed.
dataset     = EasyDict(tfrecord_dir='TrainingData')         # Options for dataset.load_dataset(). dataset is from 'TrainingData' folder of data_dir 
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
#----------------------------------------------------------------------------

###########################
# Set the following parameters according to user-defined applications.
###########################

#----------------------------------------------------------------------------
# Uncomment the following lines to further train the pre-trained GANs
#train.resume_run_id = '/scratch/users/suihong/2D_channels/TrainedModels/117-prog2D_uncd_2gpu/network-snapshot-011841.pkl'    # Run ID or network pkl to resume training from, None = start from scratch.
#train.resume_kimg = 11841  # Assumed training progress at the beginning. Affects reporting and training schedule.
#train.resume_time = 4*3600 # seconds, Assumed wallclock time at the beginning. Affects reporting.
#----------------------------------------------------------------------------
desc += '_2gpu'                      # Supplement descriptions onto the folder name of results
num_gpus = 2                         # number of gpus used for training
train.total_kimg = 900000            # thousands of training data used before stopping
G.facies_codes = [0, 1, 3, 4]        # facies code value in training dataset
G.prior_codes = [3, 4, 1, 0]         # facies codes with decreasing priority when downsampling
G.facies_indic  = True               # Decide whether the output is one facies model or several indicator models for each facies.
if not G.facies_indic: G.beta = 6e3  # Used in soft-argmax function, to be tuned for specific cases; 8e3 works ok for a 3D case where we have 3 facies: channel, lobe, and mud. If the final trained generator produces isolated facies volumes, especially isolated single-pixel facies around the surface of another facies, try to increase beta several times; G is very sensitive to beta, so beta should not be set too large in which case vanishing gradient may occur (I haven't tested). Not used when G.output_facies_mode = True
# Setting if loss normalization (into standard Gaussian) is used, if used, remember to set mean and std for each loss in loss.py  
G_loss.lossnorm = True
# size of input latent vectors. x y dim should be the same in this GANSim version
G.latent_img_num      = 8             # Number of input latent imgs.
G.latent_size_x       = 4
G.latent_size_y       = 4

D.facies_indic = G.facies_indic
G_loss.facies_indic = G.facies_indic
D_loss.facies_indic = G.facies_indic
train.facies_indic = G.facies_indic
D.facies_codes = G.facies_codes 
G_loss.facies_codes = G.facies_codes 
D_loss.facies_codes = G.facies_codes 
train.facies_codes = G.facies_codes  
G_loss.prior_codes = G.prior_codes
G.num_facies = len(G.facies_codes)            # Number of facies types.
dataset.data_range = [min(G.facies_codes), max(G.facies_codes)]

#----------------------------------------------------------------------------
# Paths.
data_dir    = '/scratch/users/suihong/Mud_drape_Channels/PreparedDataset/'  # Training data path
result_dir  = '/scratch/users/suihong/Mud_drape_Channels/TrainedModels/'  # result data path
#----------------------------------------------------------------------------
# settings for schedual of training; at different resolution levels, for progressive training method
sched.minibatch_dict           = {4: 128, 8: 128, 16: 128, 32: 128, 64: 128}  
sched.G_lrate_dict             = {4: 0.0025, 8: 0.0025, 16: 0.0025, 32: 0.0025, 64: 0.0025} 
sched.D_lrate_dict             = sched.G_lrate_dict
sched.lod_training_kimg_dict   = {4: 320, 8:640, 16:640, 32:960, 64:1280}  # e.g., 8:1280 means 1280k imgs are used when 8 transitions to 16
sched.lod_transition_kimg_dict = {4: 640, 8:640, 16:960, 32:1280, 64:1280}  
sched.max_minibatch_per_gpu    = {32: 256, 64: 128}
sched.tick_kimg_dict           = {4: 160, 8:320, 16:320, 32:480, 64:320} 

#----------------------------------------------------------------------------
G_loss.orig_weight = 60.   # weight for original Wasserstein GAN loss
if G_loss.lossnorm: G_loss.GANloss_mean = -4.7    # used to normalize GAN loss into a Gaussian-like range;
if G_loss.lossnorm: G_loss.GANloss_std = 3.1    

#----------------------------------------------------------------------------
# Settings for condition to global features (labels)
cond_label                = True   # Whether condition to labels (global features)
labeltypes                = ['pb_cf_prop', 'fp_prop']  # point bar-to-channel fill ratio, floodplain proportion; with the same order as in the training global features, could be only one global feature but need to be consistent with the training global features.
G_loss.fpprop_weight  = 2.   # weight for the two global features in the generator loss
G_loss.pb_cf_weight  = 1.5
D_loss.fpprop_weight  = G_loss.fpprop_weight
D_loss.pb_cf_weight  = G_loss.pb_cf_weight

if cond_label: desc += '-CondLabel'  
    
if G_loss.lossnorm: 
    G_loss.fpprop_mean = 1.78    # mean and standard deviation for floodplain proportion loss
    G_loss.fpprop_std = 1.13
    G_loss.pb_cf_mean = 1.10    # mean and standard deviation for point bar-to-channel fill ratio
    G_loss.pb_cf_std = 0.41

label_size = len(labeltypes) if cond_label else 0
dataset.cond_label        = cond_label   
train.cond_label          = cond_label
G.cond_label              = cond_label
G_loss.cond_label         = cond_label
D_loss.cond_label         = cond_label
G.label_size              = label_size
dataset.labeltypes        = labeltypes
G_loss.labeltypes         = labeltypes
D_loss.labeltypes         = labeltypes

#----------------------------------------------
# Settings for condition to well facies data
cond_well                    = True    # Whether condition to well facies data
G_loss.Wellfaciesloss_weight = 100.    # Weight for well facies loss
if cond_well: desc += '-CondWell'   
if G_loss.lossnorm: G_loss.wellloss_mean = 0.0023    # mean and standard deviation for well facies loss
if G_loss.lossnorm: G_loss.wellloss_std = 0.0067
dataset.well_enlarge = False                # set for whether sparse well facies data are enlarged, e.g., well point occupies 2x2 cells instead of 1x1 cell
if cond_well and dataset.well_enlarge: desc += '-Enlarg'  # description of well enlargement onto the folder name of result.

if cond_well: 
    G_loss.global_weight = 0.08     # weight for global discriminator
    G_loss.local_weight = 1.- G_loss.global_weight   # weight for local discriminator
    D_loss.global_weight = G_loss.global_weight
    D_loss.local_weight = G_loss.local_weight
    
dataset.cond_well            = cond_well
train.cond_well              = cond_well
G.cond_well                  = cond_well
D.cond_well                  = cond_well
G_loss.cond_well             = cond_well
D_loss.cond_well             = cond_well

#----------------------------------------------
# Settings for condition to probability data
cond_prob                   = True    # Whether condition to facies probability maps/cubes
code_prob = [1, 3, 4]   # codes of probability maps taken into the generator in the same order, should be 1 code less than facies_codes
G_loss.Probimgloss_weight  = 2   # Weight for probability loss
G_loss.batch_multiplier    = 4   # How many facies models are used to calculate a frequency map to compare with input probability maps in loss.
if G_loss.lossnorm: G_loss.probloss_mean = 9005  # mean and standard deviation of probability loss
if G_loss.lossnorm: G_loss.probloss_std = 1393
if cond_prob: desc += '-CondProb' 

dataset.cond_prob           = cond_prob
train.cond_prob             = cond_prob
G.cond_prob                 = cond_prob
G_loss.cond_prob            = cond_prob
D_loss.cond_prob            = cond_prob
G_loss.code_prob_order      = [G.facies_codes.index(i) for i in code_prob]

#----------------------------------------------
# Set if no growing, i.e., the conventional training method.
#desc += '-nogrowing'; 
#sched.lod_training_kimg_dict   = {4: 0, 8:0, 16:0, 32:0, 64:0}
#sched.lod_transition_kimg_dict = {4: 0, 8:0, 16:0, 32:0, 64:0}

