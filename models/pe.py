import argparse
import logging
import os
import numpy as np
import imageio
from torchvision.utils import make_grid
import torchvision
import torch
import scipy.optimize
from scipy.optimize import root_scalar
import scipy.stats
from models.PE.pe.feature_extractor import extract_features
from models.PE.pe.metrics import make_fid_stats
from models.PE.pe.metrics import compute_fid
from models.PE.pe.dp_counter import dp_nn_histogram
from models.PE.pe.arg_utils import str2bool
from models.PE.apis import get_api_class_from_name
import torch.nn.functional as F

import logging


from models.synthesizer import DPSynther


def get_noise_multiplier(epsilon, num_steps, delta, min_noise_multiplier=1e-1, max_noise_multiplier=500, max_epsilon=1e7):

    def delta_Gaussian(eps, mu):
        # Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        if mu == 0:
            return 0
        if np.isinf(np.exp(eps)):
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)

    def eps_Gaussian(delta, mu):
        # Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return root_scalar(f, bracket=[0, max_epsilon], method='brentq').root

    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)

    def objective(x):
        return compute_epsilon(noise_multiplier=x, num_steps=num_steps, delta=delta) - epsilon

    output = root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method='brentq')

    if not output.converged:
        raise ValueError("Failed to converge")

    return output.root


class PE(DPSynther):
    def __init__(self, config, device):
        """
        Initializes the class instance with the given configuration and device.
        
        Parameters:
        - config: Configuration object containing settings for the API and other parameters.
        - device: Device on which to run the computations (e.g., 'cpu', 'cuda').
        """
        super().__init__()  # Call the constructor of the parent class
        api_class = get_api_class_from_name(config.api)  # Get the API class based on the name provided in the config
        if config.api == 'sd':
            self.api = api_class.from_dict_args(config.api_params)
        else:
            api_args = []  # Initialize an empty list to hold API arguments
            for k in config.api_params:  # Iterate over the API parameters in the configuration
                api_args.append('--' + k)  # Append the parameter key as a command-line argument
                api_args.append(str(config.api_params[k]))  # Append the parameter value as a command-line argument
            self.api = api_class.from_command_line_args(api_args)  # Initialize the API with the constructed command-line arguments
        self.feature_extractor = config.feature_extractor  # Set the feature extractor from the configuration
        self.samples = None  # Initialize the samples attribute to None
        self.labels = None  # Initialize the labels attribute to None
        self.api_params = config.api_params


    def train(self, sensitive_dataloader, config):
        # Create a directory to store logs
        os.makedirs(config.log_dir, exist_ok=True)
        tmp_folder = config.tmp_folder

        # Calculate the noise multiplier for differential privacy
        self.noise_factor = get_noise_multiplier(
            epsilon=config.dp.epsilon, 
            delta=config.dp.delta, 
            num_steps=len(config.num_samples_schedule) - 1
        )

        # Log the calculated noise factor
        logging.info("The noise factor is {}".format(self.noise_factor))

        # Initialize lists to store private samples and labels
        all_private_samples = []
        all_private_labels = []

        # Iterate over the sensitive data loader to collect samples and labels
        for x, y in sensitive_dataloader:
            if len(y.shape) == 2:
                x = x.to(torch.float32) / 255.
                y = torch.argmax(y, dim=1)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            all_private_samples.append(x.cpu().numpy())
            all_private_labels.append(y.cpu().numpy())

        # Concatenate all collected samples and labels
        all_private_samples = np.concatenate(all_private_samples)
        all_private_labels = np.concatenate(all_private_labels)

        # Clip and round the pixel values
        all_private_samples = np.around(np.clip(all_private_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
        all_private_samples = np.transpose(all_private_samples, (0, 2, 3, 1))

        # Get unique classes and their count
        private_classes = list(sorted(set(list(all_private_labels))))
        private_num_classes = len(private_classes)

        # Log the start of feature extraction
        logging.info('Extracting features')
        all_private_features = extract_features(
            data=all_private_samples,
            tmp_folder=tmp_folder,
            model_name=self.feature_extractor,
            num_workers=2,
            res=config.private_image_size,
            batch_size=config.feature_extractor_batch_size
        )
        logging.info(f'all_private_features.shape: {all_private_features.shape}')

        # Log the start of generating initial samples
        logging.info('Generating initial samples')
        labels = None

        # Generate initial samples using the API
        if 'initial_sample' in config:
            syn = np.load(config.initial_sample)
            samples, additional_info = syn["x"][:config.num_samples_schedule[0]], syn["y"][:config.num_samples_schedule[0]]
            
            samples_tensor = torch.Tensor(samples)
            print("Original samples shape:", samples_tensor.shape)

            if samples_tensor.dim() == 3:
                samples_tensor = samples_tensor.unsqueeze(1)
            elif samples_tensor.dim() != 4:
                raise ValueError(f"Unexpected samples shape: {samples_tensor.shape}")

            if samples_tensor.shape[1] == 1:
                samples_tensor = samples_tensor.repeat(1, 3, 1, 1)
                print("Converted grayscale to RGB, new shape:", samples_tensor.shape)

            model_image_size = self.api_params.model_image_size if 'model_image_size' in self.api_params else self.api_params.network.image_size
            samples_tensor = F.interpolate(
                samples_tensor,
                size=[model_image_size, model_image_size],
                mode='bilinear',
                align_corners=False
            ).clamp(0., 1.)

            samples = np.around(np.clip((samples_tensor.numpy() * 255.), a_min=0, a_max=255)).astype(np.uint8)
            samples = samples.transpose(0, 2, 3, 1)
            original_samples = np.copy(samples)
            # Add variant dimension to original_samples
            original_samples = np.expand_dims(original_samples, axis=1)

            logging.info(f'samples.shape after transpose: {samples.shape}')
            logging.info(f'original_samples.shape after expand_dims: {original_samples.shape}')
            if samples.ndim != 4 or original_samples.ndim != 5:
                raise ValueError(f"Expected samples to have 4 dims and original_samples to have 5 dims, got shapes {samples.shape}, {original_samples.shape}")
        else:
            samples, additional_info = self.api.image_random_sampling(
                prompts=config.initial_prompt,
                num_samples=config.num_samples_schedule[0],
                size=config.image_size,
                labels=labels
            )

        start_t = 1

        # Main training loop
        for t in range(start_t, len(config.num_samples_schedule)):
            logging.info(f't={t}')
            assert samples.shape[0] % private_num_classes == 0
            num_samples_per_class = samples.shape[0] // private_num_classes

            if t == len(config.num_samples_schedule) - 1:
                samples = np.concatenate([samples, original_samples[:, 0]], axis=0)
                labels = [np.array([class_i] * num_samples_per_class) for class_i, class_ in enumerate(private_classes)]
                labels.append(additional_info)
                labels = np.concatenate(labels, axis=0)
                sub_packed_features = extract_features(
                    data=samples,
                    tmp_folder=tmp_folder,
                    num_workers=2,
                    model_name=self.feature_extractor,
                    res=config.private_image_size,
                    batch_size=config.feature_extractor_batch_size
                )
                count = []
                for class_i, class_ in enumerate(private_classes):
                    sub_count, sub_clean_count = dp_nn_histogram(
                        public_features=sub_packed_features[labels==class_i],
                        private_features=all_private_features[all_private_labels == class_],
                        noise_multiplier=self.noise_factor,
                        num_nearest_neighbor=config.num_nearest_neighbor,
                        mode=config.nn_mode,
                        threshold=config.count_threshold
                    )
                    count.append(sub_count)
                # count = np.concatenate(count)

                new_num_samples_per_class = config.num_samples_schedule[t] // private_num_classes
                new_indices = []
                for class_i in private_classes:
                    sub_count = count[class_i]
                    sub_new_indices = np.random.choice(
                        np.where(labels==class_i)[0],
                        size=new_num_samples_per_class,
                        p=sub_count / np.sum(sub_count)
                    )
                    new_indices.append(sub_new_indices)
                new_indices = np.concatenate(new_indices)
                samples = samples[new_indices]
                labels = labels[new_indices]

                self.samples = np.transpose(samples.astype('float'), (0, 3, 1, 2)) / 255.
                self.labels = labels
                return

            if config.lookahead_degree == 0:
                packed_samples = np.expand_dims(samples, axis=1)
            else:
                logging.info('Running image variation')
                packed_samples = self.api.image_variation(
                    images=samples,
                    additional_info=additional_info,
                    num_variations_per_image=config.lookahead_degree,
                    size=config.image_size,
                    variation_degree=config.variation_degree_schedule[t]
                )

            # Check packed_samples shape
            logging.info(f'packed_samples.shape: {packed_samples.shape}')
            if packed_samples.ndim != 5:
                raise ValueError(f"Expected packed_samples to have 5 dims, got shape {packed_samples.shape}")

            # NEW: Check data difference between packed_samples and original_samples
            if 'initial_sample' in config:
                pixel_diff = np.mean(np.abs(packed_samples[:, 0] - original_samples[:, 0]))
                logging.info(f'Mean pixel difference between packed_samples and original_samples: {pixel_diff:.4f}')

            # Extract features from the generated samples
            packed_features = []
            logging.info('Running feature extraction')
            for i in range(packed_samples.shape[1]):
                data = packed_samples[:, i]
                logging.info(f'packed_samples[:, {i}].shape: {data.shape}')
                if data.ndim != 4:
                    raise ValueError(f"Expected packed_samples[:, {i}] to have 4 dims, got shape {data.shape}")
                sub_packed_features = extract_features(
                    data=data,
                    tmp_folder=tmp_folder,
                    num_workers=2,
                    model_name=self.feature_extractor,
                    res=config.private_image_size,
                    batch_size=config.feature_extractor_batch_size
                )
                logging.info(f'sub_packed_features.shape: {sub_packed_features.shape}')
                packed_features.append(sub_packed_features)
            packed_features = np.mean(packed_features, axis=0)

            if 'initial_sample' in config:
                logging.info('Running original feature extraction')
                logging.info(f'original_samples.shape: {original_samples.shape}')
                if original_samples.ndim != 5:
                    raise ValueError(f"Expected original_samples to have 5 dims, got shape {original_samples.shape}")
                original_packed_features = []
                for i in range(original_samples.shape[1]):
                    data = original_samples[:, i]
                    logging.info(f'original_samples[:, {i}].shape: {data.shape}')
                    if data.ndim != 4:
                        raise ValueError(f"Expected original_samples[:, {i}] to have 4 dims, got shape {data.shape}")
                    sub_original_packed_features = extract_features(
                        data=data,
                        tmp_folder=tmp_folder,
                        num_workers=2,
                        model_name=self.feature_extractor,
                        res=config.private_image_size,
                        batch_size=config.feature_extractor_batch_size
                    )
                    logging.info(f'sub_original_packed_features.shape: {sub_original_packed_features.shape}')
                    original_packed_features.append(sub_original_packed_features)
                original_packed_features = np.mean(original_packed_features, axis=0)

                # Combine packed_features and original_packed_features
                logging.info('Combining packed and original features')
                combined_features = np.concatenate([packed_features, original_packed_features], axis=0)
                packed_indices = np.arange(packed_features.shape[0])
                original_indices = np.arange(packed_features.shape[0], combined_features.shape[0])

            # Compute histograms for each class
            logging.info('Computing histogram')
            count = []
            for class_i, class_ in enumerate(private_classes):
                if 'initial_sample' in config and False:
                    packed_features_i = packed_features[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)]
                    original_packed_features_i = original_packed_features[additional_info == class_]
                    combined_feat = np.concatenate([packed_features_i, original_packed_features_i], axis=0)
                    sub_count, sub_clean_count = dp_nn_histogram(
                        public_features=combined_feat,
                        private_features=all_private_features[all_private_labels == class_],
                        noise_multiplier=self.noise_factor,
                        num_nearest_neighbor=config.num_nearest_neighbor,
                        mode=config.nn_mode,
                        threshold=config.count_threshold
                    )
                    count.append(sub_count)
                    # sub_count, sub_clean_count = dp_nn_histogram(
                    #     public_features=combined_features[
                    #         num_samples_per_class * class_i:num_samples_per_class * (class_i + 1) * 2
                    #     ],
                    #     private_features=all_private_features[all_private_labels == class_],
                    #     noise_multiplier=self.noise_factor,
                    #     num_nearest_neighbor=config.num_nearest_neighbor,
                    #     mode=config.nn_mode,
                    #     threshold=config.count_threshold
                    # )
                    # count.append(sub_count)

                    packed_count_sum = np.sum(sub_count[:len(packed_features_i)])
                    original_count_sum = np.sum(sub_count[len(packed_features_i):])
                    # packed_count_sum = np.sum(sub_count[:num_samples_per_class])
                    # original_count_sum = np.sum(sub_count[num_samples_per_class:])
                    logging.info(f'Class {class_}: packed_count_sum={packed_count_sum:.2f}, original_count_sum={original_count_sum:.2f}')
                    # Original (replaced):
                    # sub_packed_indices = packed_indices[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)]
                    # sub_original_indices = original_indices[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)]
                    # packed_count_sum = np.sum(sub_count[sub_packed_indices % num_samples_per_class])
                    # original_count_sum = np.sum(sub_count[sub_original_indices % num_samples_per_class])
                else:
                    sub_count, sub_clean_count = dp_nn_histogram(
                        public_features=packed_features[
                            num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)],
                        private_features=all_private_features[all_private_labels == class_],
                        noise_multiplier=self.noise_factor,
                        num_nearest_neighbor=config.num_nearest_neighbor,
                        mode=config.nn_mode,
                        threshold=config.count_threshold
                    )
                    count.append(sub_count)
            count = np.concatenate(count)

            # Visualize the results for each class
            for class_i, class_ in enumerate(private_classes):
                visualize(
                    samples=samples[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)],
                    packed_samples=packed_samples[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)],
                    count=count[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)],
                    folder=f'{config.log_dir}/samples',
                    suffix=f'class{class_}'
                )

            # Generate new indices based on the computed histograms
            logging.info('Generating new indices')
            assert config.num_samples_schedule[t] % config.private_num_classes == 0
            new_num_samples_per_class = config.num_samples_schedule[t] // private_num_classes
            new_indices = []
            for class_i in private_classes:
                sub_count = count[num_samples_per_class * class_i:num_samples_per_class * (class_i + 1)]
                sub_new_indices = np.random.choice(
                    np.arange(num_samples_per_class * class_i, num_samples_per_class * (class_i + 1)),
                    size=new_num_samples_per_class,
                    p=sub_count / np.sum(sub_count)
                )
                new_indices.append(sub_new_indices)
            new_indices = np.concatenate(new_indices)
            new_samples = samples[new_indices]
            additional_info = additional_info[new_indices]

            # Generate new samples based on the selected indices
            logging.info('Generating new samples')
            samples = self.api.image_variation(
                images=new_samples,
                additional_info=additional_info,
                num_variations_per_image=1,
                size=config.image_size,
                variation_degree=config.variation_degree_schedule[t]
            )
            samples = np.squeeze(samples, axis=1)

            # Log the final samples if this is the last iteration
            if t == len(config.num_samples_schedule) - 1:
                log_samples(
                    samples=samples,
                    additional_info=additional_info,
                    folder=f'{config.log_dir}/{t}',
                    plot_images=False
                )

        # Store the final samples and labels
        self.samples = np.transpose(samples.astype('float'), (0, 3, 1, 2)) / 255.
        self.labels = np.concatenate([[cls] * num_samples_per_class for cls in private_classes])


    def generate(self, config):
        # Create a directory to store logs and generated data
        os.mkdir(config.log_dir)
        
        # Interpolate the sample data to the specified resolution and convert it back to numpy array
        syn_data = F.interpolate(torch.from_numpy(self.samples), size=[config.resolution, config.resolution]).numpy()
        
        # Assign the labels for the synthetic data
        syn_labels = self.labels
        
        # Save the synthetic data and labels as an .npz file in the log directory
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        # Prepare a list of images to display
        show_images = []
        
        # Determine the number of unique classes in the labels
        num_class = len(set(list(syn_labels)))
        
        # For each class, select up to 8 samples and add them to the list of images to display
        for cls in range(num_class):
            show_images.append(syn_data[syn_labels == cls][:8])
        
        # Concatenate all selected images into a single array
        show_images = np.concatenate(show_images)
        
        # Save the concatenated images as a grid image in the log directory
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        
        # Return the synthetic data and labels
        return syn_data, syn_labels



def log_samples(samples, additional_info, folder, plot_images):
    """
    Logs the samples and additional information to a specified folder.
    
    :param samples: The sample data to be logged.
    :param additional_info: Additional information to be saved alongside the samples.
    :param folder: The directory where the files will be saved.
    :param plot_images: A boolean flag indicating whether to save images of the samples.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save the samples and additional information to a .npz file
    np.savez(
        os.path.join(folder, 'samples.npz'),
        samples=samples,
        additional_info=additional_info)
    
    # If plot_images is True, save each sample as an image
    if plot_images:
        for i in range(samples.shape[0]):
            imageio.imwrite(os.path.join(folder, f'{i}.png'), samples[i])

def log_count(count, clean_count, path):
    """
    Logs the count and clean count data to a specified path.
    
    :param count: The count data to be logged.
    :param clean_count: The clean count data to be logged.
    :param path: The file path where the data will be saved.
    """
    # Create the directory if it does not exist
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # Save the count and clean count data to a .npz file
    np.savez(path, count=count, clean_count=clean_count)

def visualize(samples, packed_samples, count, folder, suffix=''):
    """
    Visualizes the top and bottom 5 samples based on their count and saves the visualization as images.
    
    :param samples: The sample data to be visualized.
    :param packed_samples: Packed sample data for visualization.
    :param count: The count data used to determine the top and bottom samples.
    :param folder: The directory where the visualization images will be saved.
    :param suffix: An optional suffix to add to the saved image filenames.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Transpose the samples and packed_samples to the correct format for visualization
    samples = samples.transpose((0, 3, 1, 2))
    packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))
    
    # Get the indices of the top 5 samples based on count
    ids = np.argsort(count)[::-1][:5]
    print(count[ids])
    
    # Prepare the top 5 samples for visualization
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    
    # Create a grid of images for the top 5 samples
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1,
        padding=0).numpy().transpose((1, 2, 0))
    
    # Convert the image to uint8 format and save it
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_top_{suffix}.png'), vis_samples)
    
    # Get the indices of the bottom 5 samples based on count
    ids = np.argsort(count)[:5]
    print(count[ids])
    
    # Prepare the bottom 5 samples for visualization
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    
    # Create a grid of images for the bottom 5 samples
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1,
        padding=0).numpy().transpose((1, 2, 0))
    
    # Convert the image to uint8 format and save it
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_bottom_{suffix}.png'), vis_samples)

def round_to_uint8(image):
    """
    Rounds the image data to the nearest integer and clips the values to the range [0, 255], then converts to uint8.
    
    :param image: The image data to be converted.
    :return: The converted image data in uint8 format.
    """
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)
