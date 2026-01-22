from metrics.dp_metrics import DPMetric
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import os
import shutil
import glob

class DPGradient(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

        self.n_dim = 12
        self.max_images = 2000
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
                self.image_height = image_height
                self.image_width = image_width

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

                # Random projection matrix to reduce gradient dimension to vec_size
                # Gradient has shape (3 * H * W), project to vec_size
                grad_dim = 3 * image_height * image_width
                self.grad_projection = nn.Linear(grad_dim, vec_size, bias=False)

                # Fix weights to be non-trainable
                self._fix_weights()

            def _fix_weights(self):
                """Set all parameters to non-trainable."""
                for param in self.parameters():
                    param.requires_grad = False

            def forward(self, input1, input2):
                if input1.shape[-1] == 28:
                    input1 = F.interpolate(input1, size=[32, 32])
                if input2.shape[-1] == 28:
                    input2 = F.interpolate(input2, size=[32, 32])
                feat1 = self.conv_block(input1).view(input1.size(0), -1)
                feat2 = self.conv_block(input2).view(input2.size(0), -1)
                output1 = self.fc(feat1)
                output2 = self.fc(feat2)

                return output1, output2

            def forward_with_gradient(self, input1, input2):
                """
                Compute gradients of network output w.r.t. input images.
                Returns gradient vectors projected to vec_size dimension.
                """
                # Resize if needed (same as forward)
                if input1.shape[-1] == 28:
                    input1 = F.interpolate(input1, size=[32, 32])
                if input2.shape[-1] == 28:
                    input2 = F.interpolate(input2, size=[32, 32])

                # Enable gradients for input
                input1 = input1.detach().requires_grad_(True)
                input2 = input2.detach().requires_grad_(True)

                # Forward pass
                feat1 = self.conv_block(input1).view(input1.size(0), -1)
                feat2 = self.conv_block(input2).view(input2.size(0), -1)
                output1 = self.fc(feat1)
                output2 = self.fc(feat2)

                # Compute scalar loss (sum of all outputs) for gradient computation
                loss1 = output1.sum()
                loss2 = output2.sum()

                # Compute gradients w.r.t. inputs
                grad1 = torch.autograd.grad(loss1, input1, create_graph=False, retain_graph=False)[0]
                grad2 = torch.autograd.grad(loss2, input2, create_graph=False, retain_graph=False)[0]

                # Flatten gradients: (batch, 3, H, W) -> (batch, 3*H*W)
                grad1_flat = grad1.view(grad1.size(0), -1)
                grad2_flat = grad2.view(grad2.size(0), -1)

                # Project to vec_size dimension using random projection
                grad1_proj = self.grad_projection(grad1_flat)
                grad2_proj = self.grad_projection(grad2_flat)

                return grad1_proj, grad2_proj

        # Instantiate and return the network
        return FixedRandomNet(self.dataloader_size, 32 if self.image_height == 28 else self.image_height, 32 if self.image_width == 28 else self.image_width, self.vec_size)

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

        self.print_stats("reduced_variants Gradient", reduced_variants)
        self.print_stats("reduced_originals Gradient", reduced_originals)

        distance = torch.norm(reduced_variants - reduced_originals, p='fro')

        return distance

    def print_stats(self, name, tensor):
            print(f"\n📈 {name} Statistics:")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std:  {tensor.std().item():.6f}")
            print(f"  Min:  {tensor.min().item():.6f}")
            print(f"  Max:  {tensor.max().item():.6f}")
            print(f"  Median: {tensor.median().item():.6f}")
            print(f"  Non-zero count: {(tensor != 0).sum().item()} / {tensor.numel()}")

    def cal_metric(self, args):

        print("🚀 Starting DPGradient calculation...")

        # args.non_DP is False when --non_DP flag is used (store_false action)
        # So apply_dp should be the same as args.non_DP (True by default, False when flag is used)
        apply_dp = args.non_DP
        print(f"🔒 DP mode: {'Enabled' if apply_dp else 'Disabled'}")

        time = self.get_time()
        save_dir = f"{args.save_dir}/{time}-{args.sensitive_dataset}-{args.public_model}"

        # Generate variations
        original_dataloader, variations_dataloader = self._image_variation(
            self.sensitive_dataset, save_dir, max_images=2000
        )
        print(f"📊 Original_images: {len(original_dataloader.dataset)}; Variations shape: {len(variations_dataloader.dataset)}")

        random_model = self.random_network().to(self.device)

        # Process dataloaders in batches
        variant_gradients = []
        original_gradients = []

        print("🔍 Computing gradients...")
        for (var_batch, _), (orig_batch, _) in zip(variations_dataloader, original_dataloader):
            var_batch = var_batch.to(self.device)
            orig_batch = orig_batch.to(self.device)
            # Compute gradients instead of forward outputs
            var_grad, orig_grad = random_model.forward_with_gradient(var_batch, orig_batch)
            variant_gradients.append(var_grad.detach())
            original_gradients.append(orig_grad.detach())

        # Concatenate gradient outputs
        variant_output = torch.cat(variant_gradients, dim=0)
        original_output = torch.cat(original_gradients, dim=0)

        print(f"Variations gradient matrix shape: {variant_output.shape}")
        print(f"Original images gradient matrix shape: {original_output.shape}")

        result = self.svd_decomposition(variant_output, original_output, self.n_dim)

        print(f"\n📊 Results:")
        print(f"   Public model: {args.public_model}")
        print(f"   Sensitive dataset: {args.sensitive_dataset}")
        if apply_dp:
            print(f"   DPGradient Score: {result}")
        else:
            print(f"   DPGradient Score (no DP): {result}")

        if self.is_delete_variations:
            try:
                if os.path.exists(save_dir):
                    original_dir = os.path.join(save_dir, 'original')
                    variation_dir = os.path.join(save_dir, 'variation')

                    def create_grid(image_dir, output_name):
                        """Create a 10x5 grid from images in directory"""
                        image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True) +
                                            glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))[:50]
                        if len(image_files) == 0:
                            return None

                        images = [Image.open(f) for f in image_files]
                        n_images = len(images)
                        cols = 10
                        rows = (n_images + cols - 1) // cols
                        img_w, img_h = images[0].size

                        grid = Image.new('RGB', (cols * img_w, rows * img_h))
                        for idx, img in enumerate(images):
                            row = idx // cols
                            col = idx % cols
                            grid.paste(img, (col * img_w, row * img_h))

                        for img in images:
                            img.close()

                        grid_path = os.path.join(save_dir, output_name)
                        grid.save(grid_path)
                        return grid_path, n_images

                    # Create grids for original and variation
                    result_orig = create_grid(original_dir, f"{args.sensitive_dataset}_{args.public_model}_original.png")
                    result_var = create_grid(variation_dir, f"{args.sensitive_dataset}_{args.public_model}_variation.png")

                    if result_orig:
                        print(f"\n📸 Saved {result_orig[1]} original images to: {result_orig[0]}")
                    if result_var:
                        print(f"📸 Saved {result_var[1]} variation images to: {result_var[0]}")

                    # Delete original and variation subdirectories
                    if os.path.exists(original_dir):
                        shutil.rmtree(original_dir)
                    if os.path.exists(variation_dir):
                        shutil.rmtree(variation_dir)
                    print(f"🗑️ Deleted image directories in: {save_dir}")
                else:
                    print(f"\nℹ️ Directory {save_dir} does not exist, no deletion needed.")
            except Exception as e:
                print(f"\n⚠️ Error processing variations: {e}")

        print("\n✅ DPGradient calculation completed!")

        return result
    

