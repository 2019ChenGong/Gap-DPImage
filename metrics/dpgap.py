from metrics.dp_metrics import DPMetric

class DPGAP(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

    def random_network(self):
        
        return 

    def svd_decomposition(self):
        pass

    def cal_metric(self):

        print("🚀 Starting DPMetric calculation...")

        # Extract real images from dataloader
        extracted_images = self.extract_images_from_dataloader(self.sensitive_dataset, self.max_images)
        print(f"📊 Extracted {len(extracted_images)} images, and extracted image shape: {extracted_images.shape}")

        # Generate variations
        original_images, variations = self._image_variation(extracted_images)
        print(f"📊 Variations shape: {variations.shape}, and Orignial shape: {original_images.shape}")

        self.image_size = variations.shape[-2]

        print(self.image_size)