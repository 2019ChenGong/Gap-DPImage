from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np

class DPGAP(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

        self.n_dim = 10

    def random_network(self):
        torch.manual_seed(42)

        class FixedRandomNet(nn.Module):
            def __init__(self, batch_size, image_height, image_width):
                super(FixedRandomNet, self).__init__()
                self.batch_size = batch_size

                # Convolutional block for feature extraction (shared for both inputs)
                self.conv_block = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()
                )

                # Calculate feature dimension dynamically based on input height and width
                # After three stride=2 convs, spatial size is reduced by 2^3=8
                output_height = image_height // 8
                output_width = image_width // 8
                self.feature_dim = 64 * output_height * output_width  # 64 channels

                # Separate fully connected layers for each input
                self.fc = nn.Sequential(
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.batch_size)  # Output N-dim vector for variations
                )

                # Fix weights to be non-trainable
                self._fix_weights()

            def _fix_weights(self):
                """Set all parameters to non-trainable."""
                for param in self.parameters():
                    param.requires_grad = False

            def forward(self, input1, input2):
                """
                Forward pass.
                input1: (N, 3, H, W) - variations
                input2: (N, 3, H, W) - original_images
                Returns: Tuple of two tensors (output1, output2), each of shape (N, N)
                """
                # Extract features
                feat1 = self.conv_block(input1).view(input1.size(0), -1)  # Shape: (N, feature_dim)
                feat2 = self.conv_block(input2).view(input2.size(0), -1)  # Shape: (N, feature_dim)

                # Process each feature set separately
                output1 = self.fc(feat1)  # Shape: (N, N)
                output2 = self.fc(feat2)  # Shape: (N, N)

                return output1, output2

        # Get batch size, height, and width from variations (assuming it's set in cal_metric)
        batch_size = self.variations.shape[0]
        image_height = self.variations.shape[1]
        image_width = self.variations.shape[2]

        # Instantiate and return the network
        return FixedRandomNet(batch_size, image_height, image_width)

    def svd_decomposition(self, variant_output, original_output, n_dim=None):
        if n_dim is None:
            n_dim = self.n_dim  # Use class-level n_dim if not specified

        # Ensure inputs are on the same device
        device = variant_output.device

        # Perform SVD on both matrices
        U1, S1, V1 = torch.svd(variant_output)
        U2, S2, V2 = torch.svd(original_output)

        # Reduce to n_dim by selecting top singular values and vectors
        n_dim = min(n_dim, variant_output.size(0), variant_output.size(1))  # Ensure n_dim <= N
        reduced_variants = U1[:, :n_dim] @ torch.diag(S1[:n_dim])  # Shape: (N, n_dim)
        reduced_originals = U2[:, :n_dim] @ torch.diag(S2[:n_dim])  # Shape: (N, n_dim)

        distance = torch.norm(reduced_variants - reduced_originals, p='fro')

        return distance

    def cal_metric(self):

        print("🚀 Starting DPMetric calculation...")

        # Extract real images from dataloader
        extracted_images = self.extract_images_from_dataloader(self.sensitive_dataset, self.max_images)
        print(f"📊 Extracted {len(extracted_images)} images, and extracted image shape: {extracted_images.shape}")

        # Generate variations
        original_images, variations = self._image_variation(extracted_images)
        print(f"📊 Variations shape: {variations.shape}, and Orignial shape: {original_images.shape}")

        self.variations = variations

        variations_tensor = torch.from_numpy(variations).float().permute(0, 3, 1, 2)

        random_model = self.random_network()

        # variant_output: torch.Size([self.max_images, self.max_images])
        with torch.no_grad():
            variant_output, original_output = random_model(variations_tensor, original_images)

        print(f"Variations output matrix shape: {variant_output.shape}")
        print(f"Original images output matrix shape: {original_output.shape}")

        result = self.svd_decomposition(variant_output, original_output, self.n_dim)

        return result

