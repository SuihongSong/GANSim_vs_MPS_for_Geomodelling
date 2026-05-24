from typing import Optional, Tuple
import os
import numpy as np

def import_generator(
    trained_model_dir: str,
    model_name: str = "network-snapshot-026560.pkl",
    network_name: str = "Gs",
    output_size: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    """Load the pretrained generator only.

    Important:
    This function does not rebuild or resize the generator. Rebuilding for a
    practical reservoir size is handled separately by
    ``build_generator_for_reservoir()``.
    """
    import pickle
    import tensorflow.compat.v1 as tf

    if tf.get_default_session() is None:
        tf.InteractiveSession()

    model_path = os.path.join(trained_model_dir, model_name)
    print("Loading pretrained generator from:")
    print(" ", model_path)

    with open(model_path, "rb") as f:
        try:
            G, D, Gs = pickle.load(f)
        except TypeError:
            f.seek(0)
            G, D, Gs = pickle.load(f, encoding="latin1")

    if network_name == "G":
        generator = G
    elif network_name == "Gs":
        generator = Gs
    else:
        raise ValueError("network_name should be 'G' or 'Gs'.")

    print("="*60)
    print("Pretrained generator loaded successfully.")
    print("Input shapes:", generator.input_shapes)
    print("Output shapes:", generator.output_shapes)

    return generator

def build_generator_for_reservoir(
    pretrained_generator,
    nx: int,
    ny: int,
    network_name: str = "Gs_for_reservoir",
):
    """Rebuild the loaded generator for a practical reservoir size.

    The pretrained generator is learned on patches. For application examples,
    this function creates the same generator architecture with larger spatial
    latent maps and copies trainable weights from the pretrained generator.

    This function is intentionally separate from ``import_generator()``.
    """
    import tensorflow.compat.v1 as tf
    import tfutil

    if tf.get_default_session() is None:
        tf.InteractiveSession()

    if nx % 16 != 0 or ny % 16 != 0:
        raise ValueError("For this generator, nx and ny should be multiples of 16.")

    latent_size_x = int(nx / 16)
    latent_size_y = int(ny / 16)

    # Reuse the exact static keyword arguments from the pretrained generator.
    # This is safer than rebuilding with a possibly different config.G.
    static_kwargs = dict(getattr(pretrained_generator, "static_kwargs", {}))
    static_kwargs.update(
        resolution_x=nx,
        resolution_y=ny,
        latent_size_x=latent_size_x,
        latent_size_y=latent_size_y,
    )

    build_func_name = getattr(pretrained_generator, "_build_func_name", "G_paper")
    func = build_func_name if "." in build_func_name else f"networks.{build_func_name}"

    device = "/gpu:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "/cpu:0"
    with tf.device(device):
        G_large = tfutil.Network(
            network_name,
            func=func,
            **static_kwargs,
        )
        G_large.copy_trainables_from(pretrained_generator)

    print("Generator rebuilt for reservoir size.")
    print(f"  Reservoir size: {nx} x {ny}")
    print("  Input shapes:", G_large.input_shapes)
    print("  Output shapes:", G_large.output_shapes)
    return G_large

def run_gen_model(
    G,
    realization_number,
    latents=None,
    wells=None,
    probs=None,
    global_features=None,
    seed=None,
    probability_default=0.2,
    feature_ranges=None,
):
    """
    Generate facies indicator realizations using the trained conditional generator.

    The generator always uses all conditioning inputs:
        1. latent vectors
        2. global features
        3. well facies data
        4. facies probability maps

    If an input is not provided:
        - latents: randomly sampled
        - global_features: randomly sampled in normalized range [-1, 1]
        - wells: zero-well maps
        - probs: uniform probability maps with value probability_default

    Returns
    -------
    facies_indicators:
        Generated facies indicators with shape [realization_number, n_facies, nx, ny].
    """

    import numpy as np

    real_num = int(realization_number)
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1. Latent input
    # ------------------------------------------------------------------
    latent_shape = list(G.input_shapes[0][1:])

    if latents is None:
        latents_plt = rng.randn(real_num, *latent_shape).astype(np.float32)
    else:
        latents_plt = np.asarray(latents, dtype=np.float32)

        if latents_plt.shape[0] == 1 and real_num > 1:
            latents_plt = np.repeat(latents_plt, real_num, axis=0)

        if latents_plt.shape[0] != real_num:
            raise ValueError("latents must have realization_number samples.")

    # ------------------------------------------------------------------
    # 2. Global feature input
    # ------------------------------------------------------------------
    # Generator expects [N, 2, H, W]
    # Channel 0 = pointbar development index
    # Channel 1 = floodplain proportion

    _, feature_channels, feature_h, feature_w = G.input_shapes[1]

    if feature_ranges is None:
        feature_ranges = {
            "pointbar_dev": (0.20, 2.00),
            "floodplain_prop": (0.50, 0.82),
        }

    if global_features is None:
        # Random normalized feature values in [-1, 1]
        features_plt = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(real_num, feature_channels, feature_h, feature_w),
        ).astype(np.float32)

    else:
        gf = np.asarray(global_features, dtype=np.float32)

        # If input is [N, 2], normalize physical values and expand to maps.
        if gf.ndim == 2:
            if gf.shape[0] == 1 and real_num > 1:
                gf = np.repeat(gf, real_num, axis=0)

            if gf.shape[0] != real_num:
                raise ValueError("global_features must have realization_number samples.")

            pointbar_min, pointbar_max = feature_ranges["pointbar_dev"]
            floodplain_min, floodplain_max = feature_ranges["floodplain_prop"]

            gf_norm = np.zeros_like(gf, dtype=np.float32)
            gf_norm[:, 0] = 2.0 * (gf[:, 0] - pointbar_min) / (pointbar_max - pointbar_min) - 1.0
            gf_norm[:, 1] = 2.0 * (gf[:, 1] - floodplain_min) / (floodplain_max - floodplain_min) - 1.0
            gf_norm = np.clip(gf_norm, -1.0, 1.0)

            features_plt = np.tile(
                gf_norm[:, :, np.newaxis, np.newaxis],
                (1, 1, feature_h, feature_w),
            ).astype(np.float32)

        # If input is already [N, 2, H, W], use it directly.
        elif gf.ndim == 4:
            if gf.shape[0] == 1 and real_num > 1:
                gf = np.repeat(gf, real_num, axis=0)

            if gf.shape[0] != real_num:
                raise ValueError("global_features must have realization_number samples.")

            if gf.shape[1] != 2:
                raise ValueError("global_features must have 2 channels.")

            pointbar_min, pointbar_max = feature_ranges["pointbar_dev"]
            floodplain_min, floodplain_max = feature_ranges["floodplain_prop"]

            features_plt = np.zeros_like(gf, dtype=np.float32)
            features_plt[:, 0] = 2.0 * (gf[:, 0] - pointbar_min) / (pointbar_max - pointbar_min) - 1.0
            features_plt[:, 1] = 2.0 * (gf[:, 1] - floodplain_min) / (floodplain_max - floodplain_min) - 1.0
            features_plt = np.clip(features_plt, -1.0, 1.0).astype(np.float32)

        else:
            raise ValueError("global_features should have shape [N, 2] or [N, 2, H, W].")

    # ------------------------------------------------------------------
    # 3. Well facies conditioning input
    # ------------------------------------------------------------------
    # Generator expects [N, 2, H, W]
    # Channel 0 = well location indicator
    # Channel 1 = well facies code
    #
    # Important:
    # For the second channel, if facies code > 1, add 1.
    # Otherwise, keep the same code.

    _, well_channels, well_h, well_w = G.input_shapes[2]

    if wells is None:
        wellfacies_plt = np.zeros((real_num, well_channels, well_h, well_w), dtype=np.float32)

    else:
        wellfacies_plt = np.asarray(wells, dtype=np.float32)

        if wellfacies_plt.ndim == 3:
            wellfacies_plt = wellfacies_plt[np.newaxis, :, :, :]

        if wellfacies_plt.shape[0] == 1 and real_num > 1:
            wellfacies_plt = np.repeat(wellfacies_plt, real_num, axis=0)

        if wellfacies_plt.shape[0] != real_num:
            raise ValueError("wells must have realization_number samples.")

        if wellfacies_plt.shape[1] != 2:
            raise ValueError("wells must have two channels: [well indicator, well facies].")

        # Revise facies codes in the second channel.
        wellfacies_plt[:, 1] = np.where(
            wellfacies_plt[:, 1] > 1,
            wellfacies_plt[:, 1] + 1,
            wellfacies_plt[:, 1],
        )

    # ------------------------------------------------------------------
    # 4. Probability-map conditioning input
    # ------------------------------------------------------------------
    # Generator expects [N, 3, H, W]

    _, prob_channels, prob_h, prob_w = G.input_shapes[3]

    if probs is None:
        prob_plt = np.full(
            (real_num, prob_channels, prob_h, prob_w),
            probability_default,
            dtype=np.float32,
        )

    else:
        prob_plt = np.asarray(probs, dtype=np.float32)

        if prob_plt.ndim == 3:
            prob_plt = prob_plt[np.newaxis, :, :, :]

        if prob_plt.shape[0] == 1 and real_num > 1:
            prob_plt = np.repeat(prob_plt, real_num, axis=0)

        if prob_plt.shape[0] != real_num:
            raise ValueError("probs must have realization_number samples.")

    # ------------------------------------------------------------------
    # 5. Run generator and return facies indicators only
    # ------------------------------------------------------------------
    facies_indicators = G.run(
        latents_plt,
        features_plt,
        wellfacies_plt,
        prob_plt,
    )

    return facies_indicators



def load_test_data(
    data_dir: str = "/content/GenAIGeomodeling/Data/GANSim_2DPointbar_Data_Model/PreparedDataset/",
    tfrecord_dir: str = "TestData",
    minibatch_size: int = 300,
    lod: int = 0,
    verbose: bool = True,
    cond_label: bool = True,
    cond_well: bool = True,
    cond_prob: bool = True,
    well_enlarge: bool = False,
    shuffle_mb: int = 0,
    prefetch_mb: int = 0,
    pointbar_dev_range: tuple = (0.2, 2.0),
    floodplain_prop_range: tuple = (0.5, 0.82),
):
    """
    Load test data for the GANSim 2D point-bar tutorial.

    Returns a dictionary containing:
        facies: cleaned test facies maps
        global_features: two global features in original value ranges
        pointbar_dev: pointbar development index
        floodplain_prop: floodplain proportion
        probs: facies probability maps
        wells: well conditioning data in generator input format
    """

    import numpy as np
    import tensorflow.compat.v1 as tf
    import dataset

    # Create TensorFlow session if needed.
    if tf.get_default_session() is None:
        tf.InteractiveSession()

    # Load test dataset.
    test_set = dataset.load_dataset(
        data_dir=data_dir,
        verbose=verbose,
        tfrecord_dir=tfrecord_dir,
        cond_label=cond_label,
        cond_well=cond_well,
        cond_prob=cond_prob,
        well_enlarge=well_enlarge,
        shuffle_mb=shuffle_mb,
        prefetch_mb=prefetch_mb,
    )

    # Read one minibatch from the test dataset.
    data_dict = test_set.get_minibatch_np(
        minibatch_size=minibatch_size,
        lod=lod,
    )

    facies_test = data_dict["real"]          # Test facies maps
    features_test = data_dict["label"]       # Normalized global features, range [-1, 1]
    probimgs_test = data_dict["prob"]        # Facies probability maps
    wellfaciesimgs_test = data_dict["well"]  # Well facies data

    # Clean facies codes if needed.
    facies_test_out = np.where(facies_test > 2, facies_test - 1, facies_test)

    # Convert normalized global features from [-1, 1] back to original ranges.
    #
    # Feature 0 = pointbar development index
    # Suggested/original range: 0.2-2.0
    pointbar_dev = features_test[:, 0]
    pointbar_dev = (pointbar_dev / 2.0 + 0.5) * (
        pointbar_dev_range[1] - pointbar_dev_range[0]
    ) + pointbar_dev_range[0]

    # Feature 1 = floodplain proportion
    # Suggested/original range: 0.5-0.82
    floodplain_prop = features_test[:, 1]
    floodplain_prop = (floodplain_prop / 2.0 + 0.5) * (
        floodplain_prop_range[1] - floodplain_prop_range[0]
    ) + floodplain_prop_range[0]

    # Combine the two global features into one array.
    # Shape: [realization_number, 2]
    # Column 0 = pointbar development index
    # Column 1 = floodplain proportion
    global_features = np.stack(
        [pointbar_dev, floodplain_prop],
        axis=1,
    ).astype(np.float32)

    # Convert well data to generator input format.
    #
    # Channel 0 = well location indicator
    # Channel 1 = well facies code
    well_loc = np.where(wellfaciesimgs_test > 0, 1, 0).astype(np.float32)
    well_facies = (wellfaciesimgs_test - 1) * well_loc
    well_facies = np.where(well_facies > 2, well_facies - 1, well_facies)
    well_facies = well_facies.astype(np.float32)

    wells_out = np.concatenate(
        [well_loc, well_facies],
        axis=1,
    )

    test_data = {
        "facies": facies_test_out.astype(np.float32),
        "global_features": global_features,
        "pointbar_dev": pointbar_dev.astype(np.float32),
        "floodplain_prop": floodplain_prop.astype(np.float32),
        "probs": probimgs_test.astype(np.float32),
        "wells": wells_out.astype(np.float32),
    }

    print("Test data loaded.")
    print("  facies:", test_data["facies"].shape)
    print("  global_features:", test_data["global_features"].shape)
    print("  pointbar_dev:", test_data["pointbar_dev"].shape)
    print("  floodplain_prop:", test_data["floodplain_prop"].shape)
    print("  probs:", test_data["probs"].shape)
    print("  wells:", test_data["wells"].shape)

    return test_data

def load_large_ground_truth(
    large_ti_path: str,
    output_size: Tuple[int, int] = (192, 192),
    case_index: int = 0,
) -> np.ndarray:
    """Load a large synthetic reference model used as hidden truth."""
    arr = np.load(large_ti_path)

    # Accept common shapes: H,W; N,H,W; N,1,H,W; N,C,H,W.
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            truth = arr[case_index % arr.shape[0], 0]
        else:
            # If indicators are stored, convert to facies.
            truth = np.argmax(arr[case_index % arr.shape[0]], axis=0)
    elif arr.ndim == 3:
        truth = arr[case_index % arr.shape[0]]
    elif arr.ndim == 2:
        truth = arr
    else:
        raise ValueError(f"Unsupported truth array shape: {arr.shape}")

    truth = _resize_2d_nearest(truth, output_size)
    return np.rint(truth).astype(np.int16)


def sample_well_data(
    truth: np.ndarray,
    n_wells: int = 12,
    seed: Optional[int] = None,
    well_length: Optional[int] = None,
) -> np.ndarray:
    """Sample sparse vertical well facies observations from the truth map.

    Output has two channels:
    channel 0 = well indicator, channel 1 = observed facies code.
    """
    rng = np.random.RandomState(seed)
    nx, ny = truth.shape
    wells = np.zeros((2, nx, ny), dtype=np.float32)

    xs = rng.choice(np.arange(nx), size=n_wells, replace=True)
    ys = rng.choice(np.arange(ny), size=n_wells, replace=True)

    for x, y in zip(xs, ys):
        wells[0, x, y] = 1.0
        wells[1, x, y] = float(truth[x, y])

    return wells


def generate_facies_probability_maps(
    geomodel_large_truth,
    resolution_x=None,
    resolution_y=None,
    sigma=5,
    max_noise_size=6,
    seed=123,
    dtype=np.float16,
):
    """
    Generate facies probability maps using the same logic as the original notebook.

    Accepted input shapes:
        [nx, ny]              # one facies map
        [1, nx, ny]           # one facies map with one channel/batch-like dimension
        [N, nx, ny]           # multiple facies maps
        [N, 1, nx, ny]        # multiple facies maps with channel dimension

    Facies codes:
        0: background / floodplain
        1: mud drape
        2: channel fill
        3: lateral accretion / point bar

    Output shape:
        [N, 3, resolution_x, resolution_y]

    Probability channel order:
        0: mud drape
        1: channel fill
        2: lateral accretion / point bar
    """

    import numpy as np
    from scipy import ndimage

    truth = np.asarray(geomodel_large_truth)

    # ------------------------------------------------------------------
    # Convert input to standard shape: [N, 1, nx, ny]
    # ------------------------------------------------------------------
    if truth.ndim == 2:
        # [nx, ny] -> [1, 1, nx, ny]
        truth = truth[None, None, :, :]

    elif truth.ndim == 3:
        if truth.shape[0] == 1:
            # [1, nx, ny] -> [1, 1, nx, ny]
            truth = truth[None, :, :, :]
        else:
            # [N, nx, ny] -> [N, 1, nx, ny]
            truth = truth[:, None, :, :]

    elif truth.ndim == 4:
        if truth.shape[1] != 1:
            raise ValueError(
                "For 4D input, geomodel_large_truth should have shape [N, 1, nx, ny]."
            )

    else:
        raise ValueError(
            "geomodel_large_truth should have shape [nx,ny], [1,nx,ny], [N,nx,ny], or [N,1,nx,ny]."
        )

    n_model = truth.shape[0]
    nx0 = truth.shape[2]
    ny0 = truth.shape[3]

    if resolution_x is None:
        resolution_x = nx0
    if resolution_y is None:
        resolution_y = ny0

    def noise(img, max_noise_size):
        img_x = img.shape[0]
        img_y = img.shape[1]
        img_noise = np.zeros(img.shape, dtype=np.float32)

        # 1) Add noise within facies-indicator = 1 regions
        noise_num_max = np.round(np.sum(img) / 40)

        if noise_num_max > 0:
            ind_indices = np.argwhere(img >= 0.9)

            if ind_indices.shape[0] > 0:
                noise_num = np.random.RandomState(seed).randint(
                    0,
                    int(noise_num_max),
                )

                if noise_num > 0:
                    noise_indices = np.random.RandomState(seed).randint(
                        0,
                        ind_indices.shape[0],
                        noise_num,
                    )
                    noise_coos = ind_indices[noise_indices]

                    noise_sizes = np.random.RandomState(seed).randint(
                        3,
                        max_noise_size,
                        noise_num,
                    )
                    noise_values = np.random.RandomState(seed).uniform(
                        -2.0,
                        1.5,
                        noise_num,
                    )

                    for i in range(noise_num):
                        x, y = noise_coos[i]
                        s = noise_sizes[i]
                        v = noise_values[i]
                        img_noise[
                            x:min(img_x, x + s),
                            y:min(img_y, y + s),
                        ] = v

        # 2) Add noise within facies-indicator = 0 regions
        noise_num_max = np.round((img_x * img_y - np.sum(img)) / 40)

        if noise_num_max > 0:
            ind_indices = np.argwhere(img <= 0.1)

            if ind_indices.shape[0] > 0:
                noise_num = np.random.RandomState(seed).randint(
                    0,
                    int(noise_num_max),
                )

                if noise_num > 0:
                    noise_indices = np.random.RandomState(seed).randint(
                        0,
                        ind_indices.shape[0],
                        noise_num,
                    )
                    noise_coos = ind_indices[noise_indices]

                    noise_sizes = np.random.RandomState(seed).randint(
                        3,
                        max_noise_size,
                        noise_num,
                    )
                    noise_values = np.random.RandomState(seed).uniform(
                        0.0,
                        1.0,
                        noise_num,
                    )

                    for i in range(noise_num):
                        x, y = noise_coos[i]
                        s = noise_sizes[i]
                        v = noise_values[i]
                        img_noise[
                            x:min(img_x, x + s),
                            y:min(img_y, y + s),
                        ] = v

        return img_noise

    geomodel_large_truth_prob = np.zeros(
        (n_model, 3, nx0, ny0),
        dtype=dtype,
    )

    for num in range(n_model):
        if num % 100 == 0:
            print(num)

        k = sigma

        img_indc_md = (truth[num, 0] == 1).astype(np.float32)
        img_indc_cf = (truth[num, 0] == 2).astype(np.float32)
        img_indc_la = (truth[num, 0] == 3).astype(np.float32)

        noise_md = noise(img_indc_md, max_noise_size)
        noise_cf = noise(img_indc_cf, max_noise_size)
        noise_la = noise(img_indc_la, max_noise_size)

        prob_md = (
            ndimage.gaussian_filter(img_indc_md, sigma=(k, k), mode="reflect")
            + ndimage.gaussian_filter(noise_md, sigma=(k, k), mode="reflect")
        )

        prob_cf = (
            ndimage.gaussian_filter(img_indc_cf, sigma=(k, k), mode="reflect")
            + ndimage.gaussian_filter(noise_cf, sigma=(k, k), mode="reflect")
        )

        prob_la = (
            ndimage.gaussian_filter(img_indc_la, sigma=(k, k), mode="reflect")
            + ndimage.gaussian_filter(noise_la, sigma=(k, k), mode="reflect")
        )

        prob_sum = prob_md + prob_cf + prob_la
        prob_sum_max = np.amax(prob_sum)

        if prob_sum_max < 0.25 and prob_sum_max > 0:
            prob_mx = np.random.RandomState(seed).rand(1) * 0.45 + 0.25
            prob_md *= prob_mx / prob_sum_max
            prob_cf *= prob_mx / prob_sum_max
            prob_la *= prob_mx / prob_sum_max

        if prob_sum_max > 0.9:
            prob_md = prob_md / prob_sum_max * 0.9
            prob_cf = prob_cf / prob_sum_max * 0.9
            prob_la = prob_la / prob_sum_max * 0.9

        geomodel_large_truth_prob[num, 0] = np.clip(prob_md, 0.02, 1.0)
        geomodel_large_truth_prob[num, 1] = np.clip(prob_cf, 0.02, 1.0)
        geomodel_large_truth_prob[num, 2] = np.clip(prob_la, 0.02, 1.0)

    geomodel_large_truth_prob = geomodel_large_truth_prob[
        :,
        :,
        :resolution_x,
        :resolution_y,
    ]

    print("Probability maps shape:", geomodel_large_truth_prob.shape)

    return geomodel_large_truth_prob



def _resize_2d_nearest(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D image using nearest-neighbor interpolation."""
    from scipy.ndimage import zoom

    out_x, out_y = _as_int_tuple(out_size)
    if img.shape[-2:] == (out_x, out_y):
        return img.copy()

    factors = (out_x / img.shape[-2], out_y / img.shape[-1])
    return zoom(img, factors, order=0)

def _as_int_tuple(size: Tuple[int, int]) -> Tuple[int, int]:
    return int(size[0]), int(size[1])

def calculate_global_features(
    truth,
    floodplain_code=0,
    channel_fill_code=2,
    pointbar_code=3,
    eps=1e-6,
):
    """
    Calculate truth-level global geological features from one facies model.

    Returned features:
        pointbar_development_index = pointbar cells / channel-fill cells
        floodplain_proportion = floodplain cells / total cells
    """

    import numpy as np

    truth = np.asarray(truth)

    if truth.ndim == 4:
        truth_2d = truth[0, 0]
    elif truth.ndim == 3:
        truth_2d = truth[0]
    elif truth.ndim == 2:
        truth_2d = truth
    else:
        raise ValueError(
            "truth should have shape [nx,ny], [1,nx,ny], [N,nx,ny], or [N,1,nx,ny]."
        )

    total_cells = truth_2d.size

    floodplain_cells = np.sum(truth_2d == floodplain_code)
    channel_fill_cells = np.sum(truth_2d == channel_fill_code)
    pointbar_cells = np.sum(truth_2d == pointbar_code)

    pointbar_development_index = pointbar_cells / float(channel_fill_cells + eps)
    floodplain_proportion = floodplain_cells / float(total_cells)

    return {
        "pointbar_development_index": float(pointbar_development_index),
        "floodplain_proportion": float(floodplain_proportion),
    }

def make_global_feature_ranges(
    truth_global_features,
    relative_variance=0.10,
    absolute_variance=None,
    min_values=None,
    max_values=None,
):
    """
    Define uncertainty ranges around truth-level global feature values.

    relative_variance=0.10 means ±10% around each truth-level feature value.
    """

    import numpy as np

    if absolute_variance is None:
        absolute_variance = {}

    if min_values is None:
        min_values = {
            "pointbar_development_index": 0.0,
            "floodplain_proportion": 0.0,
        }

    if max_values is None:
        max_values = {
            "pointbar_development_index": np.inf,
            "floodplain_proportion": 1.0,
        }

    feature_ranges = {}

    for name, value in truth_global_features.items():

        if name in absolute_variance:
            half_width = absolute_variance[name]
        else:
            half_width = abs(value) * relative_variance

        vmin = value - half_width
        vmax = value + half_width

        vmin = max(vmin, min_values.get(name, -np.inf))
        vmax = min(vmax, max_values.get(name, np.inf))

        feature_ranges[name] = (float(vmin), float(vmax))

    return feature_ranges

def make_one_spatial_feature_map(output_size, vmin, vmax, trend="west_to_east"):
    """
    Make one spatial feature map.

    output_size: (nx, ny)
    vmin, vmax: minimum and maximum feature values
    trend:
        "west_to_east"
        "east_to_west"
        "north_to_south"
        "south_to_north"
        "constant"
    """

    import numpy as np

    nx, ny = output_size

    if trend == "west_to_east":
        feature_map = np.tile(np.linspace(vmin, vmax, ny), (nx, 1))

    elif trend == "east_to_west":
        feature_map = np.tile(np.linspace(vmax, vmin, ny), (nx, 1))

    elif trend == "north_to_south":
        feature_map = np.tile(np.linspace(vmin, vmax, nx)[:, None], (1, ny))

    elif trend == "south_to_north":
        feature_map = np.tile(np.linspace(vmax, vmin, nx)[:, None], (1, ny))

    elif trend == "constant":
        feature_map = np.ones((nx, ny)) * ((vmin + vmax) / 2)

    else:
        raise ValueError("trend should be: west_to_east, east_to_west, north_to_south, south_to_north, or constant.")

    return feature_map.astype(np.float32)
