from metrics.dp_metrics import DPMetric

class DPFID(DPMetric):

    def __init__(self, sensitive_dataset, public_model, epsilon):

        super().__init__(sensitive_dataset, public_model, epsilon)

    def variant(self):

        images = self.extract_images_from_dataloader(self.sensitive_dataset)
        varied_images = self._image_variation(images)

        return varied_images

    def cal_metric(self):

        print("🚀 Starting DPMetric calculation...")

        extracted_images = self.extract_images_from_dataloader(self.sensitive_dataset, self.max_images)
        print(f"📊 Extracted {len(extracted_images)} images")
        
        variations = self._image_variation(extracted_images)
        print(f"📊 Variations shape: {variations.shape}")
        
        print("✅ DPMetric calculation completed!")
        return variations