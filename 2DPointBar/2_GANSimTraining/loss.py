import numpy as np
import tensorflow.compat.v1 as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def wellfc_downscale2d_1step(x, factor_array, prior_codes):
    # factor_array includes 2 factors to downscale variable x along x, y dimension.
    # prior_codes: facies codes with decreasing priority, e.g., [1, 2, 0]
    if np.all(factor_array == 1): return x
    prior_codes = [float(k) for k in prior_codes]
    ### downsample facies channel with facies priority
    # (1) arrange facies codes into decreasing code values based on priority: non-well cells are assigned code of -9, with-well cells are set e.g., 
    # code 1 (highest priority) -> 99, code 2 (moderate priority) -> 98, code 3 (lowest priority) -> 97.
    facies_channel = tf.where(tf.math.equal(x[:, 0:1], 0.), tf.fill(tf.shape(x[:, 0:1]), -9.), x[:, 1:2]) # shape of [N, 1, x_cells, y_cells, z_cells] e.g., [N, 1, 128, 128, 32]; non-well cells are -9
    dec_codes = [99. - i for i in range(len(prior_codes))] # e.g., [99, 98, 97]
    for i in range(len(prior_codes)):
        code = prior_codes[i]
        dec_code = dec_codes[i]
        facies_channel = tf.where(tf.math.equal(facies_channel, code), tf.fill(tf.shape(facies_channel), dec_code), facies_channel)
    # (2) use max_pool to downsample and get the maximum codes based on the priority code list
    # ksize can not be set as tensors in this environment.
    facies_channel = tf.nn.max_pool2d(facies_channel, ksize=factor_array, strides=factor_array, padding='VALID', data_format='NCHW') # shape of [N, 1, x_cells, y_cells, z_cells] e.g., [N, 1, 128, 128, 32]; non-well cells are -9
    facies_loc = tf.where(tf.math.greater(facies_channel, 0.), tf.fill(tf.shape(facies_channel), 1.), tf.fill(tf.shape(facies_channel), 0.))       
    #(3) get decreased codes back into original codes: e.g., -9 -> 0, 99->1, 98->2, 97->0
    prior_codes_ = prior_codes + [0.]
    dec_codes_ = dec_codes + [-9.]
    for i in range(len(prior_codes_)):
        code = prior_codes_[i]
        dec_code = dec_codes_[i]
        facies_channel = tf.where(tf.math.equal(facies_channel, dec_code), tf.fill(tf.shape(facies_channel), code), facies_channel)
    # (4) combine facies_loc and facies_channel  
    well_comb_ds = tf.concat([facies_loc, facies_channel], axis = 1)
    return well_comb_ds
    
# Nearest-neighbor upscaling.
def upscale2d(x, factors):
    if np.all(factors == 1): return x
    factor_x = factors[0]
    factor_y = factors[1] 
    s = tf.shape(x)
    #s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor_x, 1, factor_y])
    x = tf.reshape(x, [-1, s[1], s[2] * factor_x, s[3] * factor_y])
    return x

def indicatorize(facies_img, codes):
    ind_imgs = tf.zeros((tf.shape(facies_img)[0], 0, tf.shape(facies_img)[2], tf.shape(facies_img)[3]), tf.float32)
    for code in codes:
        ind_imgs = tf.concat([ind_imgs, tf.where(tf.math.equal(facies_img, code), tf.fill(tf.shape(facies_img), 1.), tf.fill(tf.shape(facies_img), 0.))], axis = 1)
    return ind_imgs   

def vonMisesProb(x_vector, mu, k):
    prob_vector = 1 / 2 * np.pi * np.exp(k * np.cos(2 * np.pi / 30 * (x_vector - mu)))
    prob_vector = prob_vector / np.sum(prob_vector)    
    return prob_vector
def vonMisesProbMatrix(x_vector, mus, k):
    prob_matrix = np.zeros((mus.shape[0], x_vector.shape[0]), dtype = np.float32)
    for i in range(mus.shape[0]):
        prob_matrix[i] = vonMisesProb(x_vector, mus[i], k)
    return prob_matrix     
#----------------------------------------------------------------------------
# Generator loss function.

#** Only the labels inputted into G is of the form of img (same size as latent vectors); labels from D is still of form [None, label size]


def G_wgan_acgan(G, D, lod, 
                 labels, well_facies, prob_imgs, 
                 minibatch_size, 
                 resolution_x        = 64,           # Output resolution. Overridden based on dataset.
                 resolution_y        = 64,
                 facies_codes        = [0, 1, 2],
                 prior_codes         = [1, 2, 0],# facies codes with decreasing priority when downsampling
                 code_prob_order     = [1, 2],  # e.g., [1, 2] means the probmaps correspond to the second and the third code in facies_codes.
                 facies_indic = True, # Decide wether facies model or facies indicators is produced from the generator          
                 cond_well           = False,    # Whether condition to well facies data.
                 cond_prob           = False,    # Whether condition to probability maps.
                 cond_label          = False,    # Whether condition to given global features (labels).
                 orig_weight = 1., 
                 labeltypes = None, 
                 fpprop_weight  = 2.,
                 pb_cf_weight  = 1.5,
                 Wellfaciesloss_weight = 0.7, 
                 Probimgloss_weight = 0.0000001, 
                 lossnorm = True,        # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.  
                 GANloss_mean = 11.1,    # used to normalize GAN loss into a Gaussian-like range;
                 GANloss_std = 16.,     
                 fpprop_mean = 1.78,    # mean and std for the two global features to normalize the corresponding loss into a Gaussian
                 fpprop_std = 1.13,
                 pb_cf_mean = 1.10,
                 pb_cf_std = 0.41,
                 wellloss_mean = 0.,
                 wellloss_std = 1.,
                 probloss_mean = 0.,
                 probloss_std = 1.,                 
                 batch_multiplier = 4,     # used when calculating probability loss
                 global_weight   = 1.,      # weight of global discriminator.
                 local_weight    = 0.):     # weight of local discriminator.                   
    
    if cond_prob:
        prob_imgs = tf.cast(prob_imgs, tf.float32)
        prob_imgs_lg = tf.reshape(tf.tile(tf.expand_dims(prob_imgs, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[3][1:]))   
    else:
        prob_imgs_lg = tf.zeros([0] + G.input_shapes[3][1:])
        batch_multiplier = 1
    
    if cond_label:
        labels_list = []
        for feature in ['fp_prop', 'pb_cf_prop']:
            if feature in labeltypes:
                labels_list.append(tf.random.uniform([minibatch_size, 1], minval=-1, maxval=1))
        if 'fp_prop' in labeltypes:
            fp_prop_ind_ = labeltypes.index('fp_prop')
            labels_list[fp_prop_ind_] = tf.clip_by_value(labels[:, fp_prop_ind_:fp_prop_ind_+1] + tf.random.uniform([minibatch_size, 1], minval=-0.2, maxval=0.2), -1, 1)    
        if 'pb_cf_prop' in labeltypes:
            pb_cf_ind_ = labeltypes.index('pb_cf_prop')
            labels_list[pb_cf_ind_] = tf.clip_by_value(labels[:, pb_cf_ind_:pb_cf_ind_+1] + tf.random.uniform([minibatch_size, 1], minval=-0.2, maxval=0.2), -1, 1)    
      
        labels_in = tf.concat(labels_list, axis=1)  
        labels_lg = tf.reshape(tf.tile(tf.expand_dims(labels_in, 1), [1, batch_multiplier, 1]), ([-1] + [G.input_shapes[1][1].value]))
        labels_lg_img = tf.expand_dims(tf.expand_dims(labels_lg, -1), -1)
        labels_lg_img = tf.tile(labels_lg_img, [1,1,G.input_shapes[1][-2], G.input_shapes[1][-1]])
    else: 
        labels_lg_img = tf.zeros([0] + G.input_shapes[1][1:])
        
    if cond_well:
        well_facies = tf.cast(well_facies, tf.float32)
        well_facies_lg = tf.reshape(tf.tile(tf.expand_dims(well_facies, 1), [1, batch_multiplier, 1, 1, 1]), ([-1] + G.input_shapes[2][1:]))
    else:
        well_facies_lg = tf.zeros([0] + G.input_shapes[2][1:])
        
    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
 
    fake_imgs_out = G.get_output_for(latents, labels_lg_img, well_facies_lg, prob_imgs_lg, is_training=True) # shape of [N, 3, 64, 64]  
        
    fake_scores_out_global, fake_scores_out_local_lg, fake_scores_out_local_md, fake_scores_out_local_sm, fake_labels_out = fp32(D.get_output_for(fake_imgs_out, well_facies_lg[:, :1], is_training=True))
    loss_local_lg = -fake_scores_out_local_lg 
    loss_local_md = -fake_scores_out_local_md
    loss_local_sm = -fake_scores_out_local_sm  
    loss_global = - fake_scores_out_global
    loss_local_lg = tfutil.autosummary('Loss_G/GANloss_local_lg', loss_local_lg)
    loss_local_md = tfutil.autosummary('Loss_G/GANloss_local_md', loss_local_md)
    loss_local_sm = tfutil.autosummary('Loss_G/GANloss_local_sm', loss_local_sm)
    loss_global = tfutil.autosummary('Loss_G/GANloss_global', loss_global)
    
    loss = (0.2*loss_local_lg + 1.*loss_local_md + 5*loss_local_sm) * local_weight + loss_global * global_weight   
    if lossnorm: loss = (loss - GANloss_mean) / GANloss_std   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     
 
    if cond_label:  
        with tf.name_scope('LabelPenalty'):
            if 'pb_cf_prop' in labeltypes:
                def add_pb_cf_penalty(weight):
                    pb_cf_ind = labeltypes.index('pb_cf_prop')
                    pb_cfPenalty = tf.nn.l2_loss(labels_lg[:, pb_cf_ind] - fake_labels_out[:, pb_cf_ind]) 
                    if lossnorm: pb_cfPenalty = (pb_cfPenalty -pb_cf_mean) / pb_cf_std  # To normalize this loss 
                    pb_cfPenalty = tfutil.autosummary('label_penalty_G/pb_cfPenalty', pb_cfPenalty)  
                    return loss + pb_cfPenalty * weight
                loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 3.5)), lambda: add_pb_cf_penalty(pb_cf_weight), lambda: loss)  # 2.0     
            if 'fp_prop' in labeltypes:
                def add_fpprop_penalty(weight):
                    fpprop_ind = labeltypes.index('fp_prop')
                    fppropPenalty = tf.nn.l2_loss(labels_lg[:, fpprop_ind] - fake_labels_out[:, fpprop_ind]) 
                    if lossnorm: fppropPenalty = (fppropPenalty -fpprop_mean) / fpprop_std  # To normalize this label_penalty 
                    fppropPenalty = tfutil.autosummary('label_penalty_G/fppropPenalty', fppropPenalty)    
                    return loss + fppropPenalty * weight
                loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 3.6)), lambda: add_fpprop_penalty(fpprop_weight), lambda: loss)  # 3.5
                  
    if cond_well:   
        def Wellpoints_L2loss(well_facies, fake_imgs):
            # Theoretically, the easiest way for downsampling well facies data is, I) get lod value from lod tensor,
            # II) calculate the downsampling factor_array as did in networks.py, III) downsampling, IV) upsampling into original resolution.
            # However, the question lies in, how to get the lod value from lod tensor. Here, we use tf1, tf.disable_v2_behavior(), and tf.disable_eager_execution(),
            # in such environment, tf.cond() does not work as expected, I tried many ways to work around tf.cond() but sill not succeed. 
            # Thus, have to iteratively downsample step by step. Also, tf.nn.max_pool3d can not use tensors as the kernels...
            # 1. well facies --downsample-with-priority--> upsample back into original resolution               
            out_sizes_log2 = np.array(np.log2([resolution_x, resolution_y]).astype(int))  # [8, 8]
            out_sizes_log2_lg = max(out_sizes_log2)  # 8
            well_facies_dsmp = well_facies
            for i in range(out_sizes_log2_lg-1):
                i_tf = tf.cast(tf.fill([], i), tf.float32)  
                dw_fct = out_sizes_log2 - i - 1
                dw_fct = np.where(dw_fct >= np.array([2, 2]), 2, 1)   # assume the coarest res is 4x4, corresponding to 2, 2              
                well_facies_dsmp = tf.cond(tf.math.less(i_tf, tf.floor(lod)), lambda: wellfc_downscale2d_1step(well_facies_dsmp, dw_fct, prior_codes), lambda: well_facies_dsmp)              
            well_facies = well_facies_dsmp
            
            well_facies_upsmp = well_facies  
            for j in range(out_sizes_log2_lg-1):
                j_tf = tf.cast(tf.fill([], j), tf.float32) 
                up_fct = out_sizes_log2 - j - 1
                up_fct = np.where(up_fct >= np.array([2, 2]), 2, 1)   # assume the coarest res is 4x4, corresponding to 2, 2               
                well_facies_upsmp = tf.cond(tf.math.less(j_tf, tf.floor(lod)), lambda: upscale2d(well_facies_upsmp, up_fct), lambda: well_facies_upsmp) 
            well_facies = well_facies_upsmp
            if facies_indic:
                well_facies = tf.concat([well_facies[:,0:1], indicatorize(well_facies[:,1:2], facies_codes) * well_facies[:,0:1]], axis = 1)
            # 2. calculate loss based on difference of input well facies and output fake_imgs    
            loss = tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:] - fake_imgs))
            loss = loss / tf.reduce_sum(well_facies[:, 0:1])
            return loss
        def addwellfaciespenalty(well_facies, fake_imgs_out, loss, Wellfaciesloss_weight):
            with tf.name_scope('WellfaciesPenalty'):
                WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_imgs_out)       
                if lossnorm: WellfaciesPenalty = (WellfaciesPenalty - wellloss_mean) / wellloss_std   # 0.002742
                WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
                loss += WellfaciesPenalty * Wellfaciesloss_weight   
            return loss   
        loss = tf.cond(tf.math.less(lod, tf.fill([], 4.)), lambda: addwellfaciespenalty(well_facies_lg, fake_imgs_out, loss, Wellfaciesloss_weight), lambda: loss)
  
    if cond_prob:
        def addprobloss(probs, fake_probs_, weight, batchsize, relzs, loss):  # fakes as indicators: [N*relzs, 3, 64, 64]        
            with tf.name_scope('ProbimgPenalty'):   
                probs_fake = tf.reduce_mean(tf.reshape(fake_probs_, ([batchsize, relzs] + G.input_shapes[3][1:])), 1)   # probs for different indicators         
                ProbPenalty = tf.nn.l2_loss(probs - probs_fake)  # L2 loss
                if lossnorm: ProbPenalty = (ProbPenalty- probloss_mean) / probloss_std   # normalize
                ProbPenalty = tfutil.autosummary('Loss_G/ProbPenalty', ProbPenalty)
            loss += ProbPenalty * weight
            return loss
        fake_probs = tf.gather(fake_imgs_out, indices=code_prob_order, axis=1) if facies_indic else tf.gather(indicatorize(fake_imgs_out, facies_codes), indices=code_prob_order, axis=1)
        loss = tf.cond(tf.math.less(lod, tf.fill([], 3.5)), lambda: addprobloss(prob_imgs, fake_probs, Probimgloss_weight, minibatch_size, batch_multiplier, loss), lambda: loss)        
     
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)    
    return loss


#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, lod, opt, minibatch_size, reals, labels, well_facies, prob_imgs, facies_codes,
    cond_well       = False,    # Whether condition to well facies data.
    cond_prob       = False,    # Whether condition to probability maps.
    cond_label      = False,    # Whether condition to given global features (labels).  
    facies_indic    = True, # Decide wether facies model or facies indicators is produced from the generator                 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.              
    labeltypes = None,  
    fpprop_weight  = 2.,
    pb_cf_weight  = 2.,
    orig_w       = 10.,
    global_weight   = 2.,      # weight of global realism term.
    local_weight    = 0.):     # weight of local realism term.               

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    
    if cond_label:
        labels_img = tf.expand_dims(tf.expand_dims(labels, -1), -1)
        labels_img = tf.tile(labels_img, [1,1,G.input_shapes[1][-2], G.input_shapes[1][-1]])   
    else:
        labels_img = tf.zeros([0] + G.input_shapes[1][1:])
    fake_imgs_out = G.get_output_for(latents, labels_img, well_facies, prob_imgs, is_training=True)  # shape of [N, 3, 64, 64]  
    reals_input = indicatorize(reals, facies_codes) if facies_indic else reals     # indicators of real facies imgs; shape of [N, 3, 64, 64] 
        
    real_scores_out_global, real_scores_out_local_lg, real_scores_out_local_md, real_scores_out_local_sm, real_labels_out = fp32(D.get_output_for(reals_input, well_facies[:, 0:1], is_training=True))
    fake_scores_out_global, fake_scores_out_local_lg, fake_scores_out_local_md, fake_scores_out_local_sm, fake_labels_out = fp32(D.get_output_for(fake_imgs_out, well_facies[:, 0:1], is_training=True))
    real_scores_out_global = tfutil.autosummary('Loss_D/real_scores_global', real_scores_out_global)
    real_scores_out_local_lg = tfutil.autosummary('Loss_D/real_scores_out_local_lg', real_scores_out_local_lg) 
    real_scores_out_local_md = tfutil.autosummary('Loss_D/real_scores_out_local_md', real_scores_out_local_md) 
    real_scores_out_local_sm = tfutil.autosummary('Loss_D/real_scores_out_local_sm', real_scores_out_local_sm) 
    fake_scores_out_global = tfutil.autosummary('Loss_D/fake_scores_out_global', fake_scores_out_global)
    fake_scores_out_local_lg = tfutil.autosummary('Loss_D/fake_scores_out_local_lg', fake_scores_out_local_lg)
    fake_scores_out_local_md = tfutil.autosummary('Loss_D/fake_scores_out_local_md', fake_scores_out_local_md)
    fake_scores_out_local_sm = tfutil.autosummary('Loss_D/fake_scores_out_local_sm', fake_scores_out_local_sm)
    
    loss_local_lg = fake_scores_out_local_lg - real_scores_out_local_lg 
    loss_local_md = fake_scores_out_local_md - real_scores_out_local_md
    loss_local_sm = fake_scores_out_local_sm - real_scores_out_local_sm
    loss_global = fake_scores_out_global - real_scores_out_global     
    
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_imgs_out.dtype)
        mixed_imgs_out = tfutil.lerp(tf.cast(reals_input, fake_imgs_out.dtype), fake_imgs_out, mixing_factors)
        mixed_scores_out_global, mixed_scores_out_local_lg, mixed_scores_out_local_md, mixed_scores_out_local_sm, mixed_labels_out = fp32(D.get_output_for(mixed_imgs_out, well_facies[:, 0:1], is_training=True))
        
        mixed_loss_local_lg = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_lg))
        mixed_grads_local_lg = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_lg, [mixed_imgs_out])[0]))
        mixed_norms_local_lg = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_lg), axis=[1,2,3]))
        gradient_penalty_local_lg = tf.square(mixed_norms_local_lg - wgan_target)
        loss_local_lg += gradient_penalty_local_lg * (wgan_lambda / (wgan_target**2))
        loss_local_lg = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_lg', loss_local_lg)

        mixed_loss_local_md = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_md))
        mixed_grads_local_md = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_md, [mixed_imgs_out])[0]))
        mixed_norms_local_md = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_md), axis=[1,2,3]))
        gradient_penalty_local_md = tf.square(mixed_norms_local_md - wgan_target)
        loss_local_md += gradient_penalty_local_md * (wgan_lambda / (wgan_target**2))
        loss_local_md = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_md', loss_local_md)

        mixed_loss_local_sm = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_local_sm))
        mixed_grads_local_sm = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_local_sm, [mixed_imgs_out])[0]))
        mixed_norms_local_sm = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_local_sm), axis=[1,2,3]))
        gradient_penalty_local_sm = tf.square(mixed_norms_local_sm - wgan_target)
        loss_local_sm += gradient_penalty_local_sm * (wgan_lambda / (wgan_target**2))
        loss_local_sm = tfutil.autosummary('Loss_D/WGAN_GP_loss_local_sm', loss_local_sm)


        mixed_loss_global = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out_global))
        mixed_grads_global = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss_global, [mixed_imgs_out])[0]))
        mixed_norms_global = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads_global), axis=[1,2,3]))
        gradient_penalty_global = tf.square(mixed_norms_global - wgan_target)
        loss_global += gradient_penalty_global * (wgan_lambda / (wgan_target**2))
        loss_global = tfutil.autosummary('Loss_D/WGAN_GP_loss_global', loss_global)
              
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty_local_lg = tfutil.autosummary('Loss_D/epsilon_penalty_local_lg', tf.square(real_scores_out_local_lg))
        loss_local_lg += epsilon_penalty_local_lg * wgan_epsilon

        epsilon_penalty_local_md = tfutil.autosummary('Loss_D/epsilon_penalty_local_md', tf.square(real_scores_out_local_md))
        loss_local_md += epsilon_penalty_local_md * wgan_epsilon

        epsilon_penalty_local_sm = tfutil.autosummary('Loss_D/epsilon_penalty_local_sm', tf.square(real_scores_out_local_sm))
        loss_local_sm += epsilon_penalty_local_sm * wgan_epsilon

        epsilon_penalty_global = tfutil.autosummary('Loss_D/epsilon_penalty_global', tf.square(real_scores_out_global))
        loss_global += epsilon_penalty_global * wgan_epsilon
    
    loss = ((0.2*loss_local_lg + 1.*loss_local_md + 5*loss_local_sm) * local_weight + loss_global * global_weight) * orig_w

    if cond_label:
        with tf.name_scope('LabelPenalty'):  
            if 'fp_prop' in labeltypes:
                def add_fpprop_penalty(weight):
                    fpprop_ind = labeltypes.index('fp_prop')
                    fppropPenalty_real = tf.nn.l2_loss(labels[:, fpprop_ind] - real_labels_out[:, fpprop_ind]) 
                    fppropPenalty_fake = tf.nn.l2_loss(labels[:, fpprop_ind] - fake_labels_out[:, fpprop_ind]) 
                    return loss + (fppropPenalty_real * 0.5 + fppropPenalty_fake * 0.5) * weight
                loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: add_fpprop_penalty(fpprop_weight), lambda: loss)  # 3.5
            if 'pb_cf_prop' in labeltypes:
                def add_pb_cf_penalty(weight):
                    pb_cf_ind = labeltypes.index('pb_cf_prop')
                    pb_cfPenalty_real = tf.nn.l2_loss(labels[:, pb_cf_ind] - real_labels_out[:, pb_cf_ind]) 
                    pb_cfPenalty_fake = tf.nn.l2_loss(labels[:, pb_cf_ind] - fake_labels_out[:, pb_cf_ind])
                    return loss + (pb_cfPenalty_real * 0.5 + pb_cfPenalty_fake * 0.5) * weight
                loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: add_pb_cf_penalty(pb_cf_weight), lambda: loss)  # 2.0

    loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss

