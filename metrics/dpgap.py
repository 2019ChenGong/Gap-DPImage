from metrics.dp_metrics import DPMetric

import torch
import torch.nn as nn
import numpy as np

class DPGAP(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

        self.n_dim = 10
        self.vec_size = self.max_images
        self.sensitive_dataset = sensitive_dataset

        for batch, _ in self.sensitive_dataset:
            self.image_height = batch.shape[2]
            self.image_width = batch.shape[3]
            break


    def random_network(self):
        torch.manual_seed(42)

        class FixedRandomNet(nn.Module):
            def __init__(self, batch_size, image_height, image_width, vec_size):
                super(FixedRandomNet, self).__init__()
                self.batch_size = batch_size
                self.vec_size = vec_size

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
                    nn.Linear(512, self.vec_size) 
                )

                # Fix weights to be non-trainable
                self._fix_weights()

            def _fix_weights(self):
                """Set all parameters to non-trainable."""
                for param in self.parameters():
                    param.requires_grad = False

            def forward(self, input1, input2):
                feat1 = self.conv_block(input1).view(input1.size(0), -1)
                feat2 = self.conv_block(input2).view(input2.size(0), -1)
                output1 = self.fc(feat1)
                output2 = self.fc(feat2)

                return output1, output2

        # Instantiate and return the network
        return FixedRandomNet(self._variation_batch_size, self.image_height, self.image_width, self.vec_size)

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

    def cal_metric(self, args):

        print("🚀 Starting DPGap calculation...")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        original_dataloader, variations_dataloader = self._image_variation(self.sensitive_dataset, save_dir)
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations shape: {len(variations_dataloader.dataset)}")

        random_model = self.random_network()

        # Process dataloaders in batches
        variant_outputs = []
        original_outputs = []
        
        with torch.no_grad():
            for (var_batch, _), (orig_batch, _) in zip(variations_dataloader, original_dataloader):
                var_batch = var_batch.to(self.device)
                orig_batch = orig_batch.to(self.device)
                var_out, orig_out = random_model(var_batch, orig_batch)
                variant_outputs.append(var_out)
                original_outputs.append(orig_out)

        # Concatenate outputs
        variant_output = torch.cat(variant_outputs, dim=0)
        original_output = torch.cat(original_outputs, dim=0)

        print(f"Variations output matrix shape: {variant_output.shape}")
        print(f"Original images output matrix shape: {original_output.shape}")

        result = self.svd_decomposition(variant_output, original_output, self.n_dim)

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)  # Recursively delete the directory and its contents
                    print(f"🗑️ Deleted directory: {save_dir}")
                else:
                    print(f"ℹ️ Directory {save_dir} does not exist, no deletion needed.")
            except Exception as e:
                print(f"⚠️ Error deleting directory {save_dir}: {e}")
            print("✅ DPGap calculation completed!")

        else:
            print("✅ DPGap calculation completed!")

        return result
    

