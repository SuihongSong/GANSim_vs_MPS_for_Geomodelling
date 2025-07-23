import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.disable_eager_execution()
import config
import tfutil
import dataset
import misc

#----------------------------------------------------------------------------
# Just-in-time processing of training imgs before feeding them to the networks. 

def process_realimgs(x, out_size_x, out_size_y):
    with tf.name_scope('ProcessRealimgs'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            x = misc.zoom_to_size(x, out_size_x, out_size_y)
        return x
    
def indicatorize(facies_img, codes):
    ind_imgs = np.zeros([facies_img.shape[0], len(codes), facies_img.shape[2], facies_img.shape[3]], np.float32)
    for i, code in enumerate(codes):
        ind_imgs[:, i:i+1] = np.where(facies_img == code, 1., 0.)
    return ind_imgs
   
def get_grid_images(ind_imgs, grid_column):
    s = ind_imgs.shape    
    ind_imgs = ind_imgs.reshape(-1, grid_column, s[1], s[2],s[3]) # shape of [2, 9, 3, 128, 128, 32]
    ind_imgs = np.transpose(ind_imgs, axes=[0, 2, 1, 3, 4])
    grid_reals = ind_imgs.reshape(-1, 1, s[2],s[3])
    return grid_reals
      
#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_training_kimg_dict  = {},# Thousands of real imgs to show before doubling the resolution.
        lod_transition_kimg_dict= {},      # Thousands of real imgs to show when fading in new layers.
        minibatch_base          = 32,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001,    # Learning rate for the generator.
        G_lrate_dict            = {},       # Resolution-specific overrides.
        D_lrate_base            = 0.001,    # Learning rate for the discriminator.
        D_lrate_dict            = {},       # Resolution-specific overrides.
        tick_kimg_base          = 1,      # Default interval of progress snapshots.
        tick_kimg_dict          = {}): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        
        train_kimg_sum = 0
        trans_kimg_sum = 0 
        
        for i in range(len(lod_training_kimg_dict)):
            train_list = list(lod_training_kimg_dict.values())
            trans_list = list(lod_transition_kimg_dict.values())
            train_kimg_sum += train_list[i]
            trans_kimg_sum += trans_list[i]

            if train_kimg_sum + trans_kimg_sum > self.kimg: 
                phase_idx = i
                lod_training_kimg = train_list[i]
                lod_transition_kimg = trans_list[i]
                break
            phase_idx = i
            lod_training_kimg = train_list[i]
            lod_transition_kimg = trans_list[i]
        phase_kimg = self.kimg - ((train_kimg_sum - train_list[phase_idx]) + (trans_kimg_sum - trans_list[i]))                

        # Level-of-detail and resolution.
        lod_initial_resolution = list(lod_training_kimg_dict.keys())[0]
        self.lod = training_set.resolution_x_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_x_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_gan(
    cond_well               = False,    # Whether condition to well facies data.
    cond_prob               = False,    # Whether condition to probability maps.
    cond_label              = False,    # Whether condition to given global features (labels).
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    facies_codes            = [0, 1, 2],    # list of facies codes
    D_repeats               = 1,            # How many times the discriminator is trained per G iteration.
    G_repeats               = 2,            # How many times the generator is trained per G iteration.
    minibatch_repeats       = 2,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 55000,        # Total length of the training, measured in thousands of real imgs.
    img_snapshot_ticks    = 1,            # How often to export img snapshots?
    network_snapshot_ticks  = 1,           # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None, #'/scratch/users/suihong/2D_channels/GANSimForGlobalFeatureConditioning/TrainedModels/047-prog_4gpu/network-snapshot-001920.pkl',        # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    facies_indic            = True,        # decide whether facies indicators are produced from the generator.
    resume_kimg             = 0,   # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0): # seconds, Assumed wallclock time at the beginning. Affects reporting.   
    
    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tfutil.Network('G', resolution_x=training_set.shape[1], resolution_y=training_set.shape[2], label_size=config.label_size, **config.G)
            D = tfutil.Network('D', resolution_x=training_set.shape[1], resolution_y=training_set.shape[2], label_size=config.label_size, **config.D)
            Gs = G.clone('Gs')
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)
    G.print_layers(); D.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        train_in_dict = training_set.get_minibatch_tf()
        reals = train_in_dict['real']
        reals_split     = tf.split(reals, config.num_gpus)
        if cond_label: 
            labels = train_in_dict['label']
            labels_split = tf.split(labels, config.num_gpus)
        if cond_prob: 
            probimgs = train_in_dict['prob']
            probimgs_split = tf.split(probimgs, config.num_gpus)
        if cond_well: 
            wellfacies = train_in_dict['well']
            wellfacies = tf.cast(wellfacies, tf.float32)
            welllocs = tf.cast((wellfacies > 0.1), tf.float32)  # obtain well locations  code 1 (mud) and code 2 (channel and levee)      
            wellfacies_corrected = (wellfacies - 1) * welllocs   #tf.where(wellfacies>2., tf.fill(wellfacies.shape, 2.), wellfacies) - 1.  # -1 to shift code into 0 and 1, for original codes are 0, 1, 2,  0 for no wells. 
            wellfacies = tf.concat([welllocs, wellfacies_corrected], 1) # now wellfacies dimension = [minibatch_in, 2, resolution, resolution] 
            wellfacies_split    = tf.split(wellfacies, config.num_gpus)
        
    G_opt = tfutil.Optimizer(name='TrainG', learning_rate=lrate_in, **config.G_opt)
    D_opt = tfutil.Optimizer(name='TrainD', learning_rate=lrate_in, **config.D_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals_gpu = process_realimgs(reals_split[gpu], training_set.shape[1], training_set.shape[2])
            labels_gpu = labels_split[gpu] if cond_label else tf.zeros([0, 0])
            wellfacies_gpu = wellfacies_split[gpu] if cond_well else tf.zeros([0] + G.input_shapes[2][1:])
            probimgs_gpu = probimgs_split[gpu] if cond_prob else tf.zeros([0] + G.input_shapes[3][1:])
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, lod = lod_in, labels = labels_gpu, well_facies = wellfacies_gpu, prob_imgs = probimgs_gpu, minibatch_size=minibatch_split, **config.G_loss)
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, lod = lod_in, opt=D_opt, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, well_facies = wellfacies_gpu, prob_imgs = probimgs_gpu, **config.D_loss)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    print('Setting up snapshot image grid...')
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config.sched) 
    grid_facies_num = 2 
    grid_size = (9, grid_facies_num * len(facies_codes)) # 6 rows times 9 columns
    grid_in_dict = training_set.get_minibatch_np(grid_facies_num * 9) # 6 rows times 8 columns
    grid_realimgs = grid_in_dict['real']
    grid_latents = misc.random_latents(grid_facies_num * 9, G) 
    if cond_label:
        grid_labels = grid_in_dict['label']
        grid_labels_img = np.expand_dims(np.expand_dims(grid_labels, axis=-1),axis=-1)
        grid_labels_img = np.tile(grid_labels_img, (1,1,G.input_shapes[1][-2],G.input_shapes[1][-1]))
    else:
        grid_labels_img = np.zeros([0] + G.input_shapes[1][1:])
    if cond_prob: 
        grid_probimgs = grid_in_dict['prob']
    else:
        grid_probimgs = np.zeros([0] + G.input_shapes[3][1:])   
    if cond_well: 
        grid_wellfacieimgs = grid_in_dict['well']
        grid_wellfaciesimg_process = np.concatenate(((grid_wellfacieimgs > 0),  (grid_wellfacieimgs - 1) * (grid_wellfacieimgs > 0)), 1)
    else: 
        grid_wellfaciesimg_process = np.zeros([0] + G.input_shapes[2][1:])
    grid_fakeimgs = Gs.run(grid_latents, grid_labels_img, grid_wellfaciesimg_process, grid_probimgs, minibatch_size=sched.minibatch//config.num_gpus)
    
    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    grid_realimgs_inds = indicatorize(grid_realimgs, facies_codes)  # get the indicators of real facies imgs, shape of [18, len(facies_codes), 128, 128]
    grid_reals = get_grid_images(grid_realimgs_inds, 9)
    misc.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'), drange=[0,1], grid_size=grid_size) #  drange was also changed into [0,1]
    grid_fakes = get_grid_images(grid_fakeimgs, 9)
    misc.save_image_grid(grid_fakes, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=[0,1], grid_size=grid_size)
    if cond_well:      
        if facies_indic:
            grid_wellfacies = indicatorize(grid_wellfaciesimg_process[:, 1:2], facies_codes) * grid_wellfaciesimg_process[:, 0:1]  # shape of [N, 3, 128, 128]
            grid_wellfacies = get_grid_images(grid_wellfacies, 9)
            misc.save_image_grid(grid_wellfacies, os.path.join(result_subdir, 'wellfacies.png'), drange=[0,1], grid_size=grid_size)
        else:
            grid_wellfacies = grid_wellfaciesimg_process[:,0:1] + grid_wellfaciesimg_process[:,1:2]
            misc.save_image_grid(grid_wellfacies/(max(facies_codes) + 1), os.path.join(result_subdir, 'wellfacies.png'), drange=[0,1], grid_size=grid_size)
    if cond_prob:
        if facies_indic:
            grid_probs = np.concatenate([1. - np.sum(grid_probimgs, axis = 1, keepdims = True), grid_probimgs], axis = 1) # shape of [N, 3, 128, 128, 32]
            grid_probs = get_grid_images(grid_probs, 9)
            misc.save_image_grid(grid_probs, os.path.join(result_subdir, 'probimages.png'), drange=[0,1], grid_size=grid_size)  
        else: 
            misc.save_image_grid(grid_probimgs[:,1], os.path.join(result_subdir, 'probimages.png'), drange=[0,1], grid_size=grid_size)  
           
    grid_latents = misc.random_latents(grid_facies_num * 9, G) 
    grid_fakeimgs = Gs.run(grid_latents, grid_labels_img, grid_wellfaciesimg_process, grid_probimgs, minibatch_size=sched.minibatch//config.num_gpus)
    grid_fakes = get_grid_images(grid_fakeimgs, 9)
    misc.save_image_grid(grid_fakes, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=[0,1], grid_size=grid_size)

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0    
    while cur_nimg < total_kimg * 1000:
        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched) 
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for repeat in range(minibatch_repeats):
            for _ in range(D_repeats):
                tfutil.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch}) #                              
            for _ in range(G_repeats):
                tfutil.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})   
            cur_nimg += sched.minibatch
 
        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', sched.lod),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            # Save snapshots.
            if cur_tick % img_snapshot_ticks == 0 or done:
                grid_fakeimgs = Gs.run(grid_latents, grid_labels_img, grid_wellfaciesimg_process, grid_probimgs, minibatch_size=sched.minibatch//config.num_gpus)
                grid_fakes = get_grid_images(grid_fakeimgs, 9)  
                misc.save_image_grid(grid_fakes, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=[0,1], grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()
            
    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.train['func'])
    tfutil.call_func_by_name(**config.train)
    print('Exiting...')

#----------------------------------------------------------------------------