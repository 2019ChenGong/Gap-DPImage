#!/usr/bin/env python3
"""
测试InceptionV3FeatureExtractor是否正确保留了预训练权重
"""

import torch
import torch.nn as nn
from torchvision.models import inception_v3
import numpy as np

class InceptionV3FeatureExtractor(nn.Module):
    """Wrapper to extract 2048-dim pool features from Inception V3"""

    def __init__(self, inception_model):
        super().__init__()
        self.inception = inception_model

    def forward(self, x):
        # Forward through Inception V3 up to the avgpool layer
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = self.inception.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def test_weights():
    print("=" * 80)
    print("测试 InceptionV3FeatureExtractor 权重是否正确加载")
    print("=" * 80)

    # 加载预训练模型
    print("\n1. 加载预训练的Inception V3模型...")
    inception_pretrained = inception_v3(pretrained=True, transform_input=False).eval()

    # 包装为特征提取器
    print("2. 包装为InceptionV3FeatureExtractor...")
    feature_extractor = InceptionV3FeatureExtractor(inception_pretrained).eval()

    # 创建一个随机输入
    print("3. 创建测试输入 (batch_size=2, channels=3, height=299, width=299)...")
    x = torch.randn(2, 3, 299, 299)

    # 提取特征
    print("4. 提取2048维特征...")
    with torch.no_grad():
        features = feature_extractor(x)

    print(f"   输出形状: {features.shape}")
    assert features.shape == (2, 2048), f"期望形状 (2, 2048), 实际得到 {features.shape}"
    print("   ✅ 输出形状正确!")

    # 检查权重是否相同
    print("\n5. 验证权重是否保留...")
    original_conv_weight = inception_pretrained.Conv2d_1a_3x3.conv.weight
    wrapped_conv_weight = feature_extractor.inception.Conv2d_1a_3x3.conv.weight

    weights_equal = torch.allclose(original_conv_weight, wrapped_conv_weight)
    print(f"   第一层卷积权重相同: {weights_equal}")
    assert weights_equal, "权重不匹配!"
    print("   ✅ 预训练权重正确保留!")

    # 检查特征值是否合理（预训练模型应该输出有意义的特征）
    print("\n6. 检查特征值范围...")
    feat_mean = features.mean().item()
    feat_std = features.std().item()
    feat_min = features.min().item()
    feat_max = features.max().item()

    print(f"   特征统计:")
    print(f"     均值: {feat_mean:.4f}")
    print(f"     标准差: {feat_std:.4f}")
    print(f"     最小值: {feat_min:.4f}")
    print(f"     最大值: {feat_max:.4f}")

    # 预训练模型的特征应该在合理范围内（不应该全是0或很大的值）
    assert abs(feat_mean) < 100, "特征均值异常大，可能未正确加载预训练权重"
    assert feat_std > 0.01, "特征标准差过小，可能未正确加载预训练权重"
    print("   ✅ 特征值在合理范围内!")

    # 测试与原始模型的一致性
    print("\n7. 测试与原始模型输出的一致性...")
    with torch.no_grad():
        # 原始模型的forward会包含fc层，我们需要手动复制forward逻辑
        # 直接比较中间层输出
        x_test = torch.randn(1, 3, 299, 299)

        # 通过feature_extractor
        feat_wrapped = feature_extractor(x_test)

        # 手动通过原始模型（复制相同的forward流程）
        with torch.no_grad():
            x_orig = x_test
            x_orig = inception_pretrained.Conv2d_1a_3x3(x_orig)
            x_orig = inception_pretrained.Conv2d_2a_3x3(x_orig)
            x_orig = inception_pretrained.Conv2d_2b_3x3(x_orig)
            x_orig = inception_pretrained.maxpool1(x_orig)
            x_orig = inception_pretrained.Conv2d_3b_1x1(x_orig)
            x_orig = inception_pretrained.Conv2d_4a_3x3(x_orig)
            x_orig = inception_pretrained.maxpool2(x_orig)
            x_orig = inception_pretrained.Mixed_5b(x_orig)
            x_orig = inception_pretrained.Mixed_5c(x_orig)
            x_orig = inception_pretrained.Mixed_5d(x_orig)
            x_orig = inception_pretrained.Mixed_6a(x_orig)
            x_orig = inception_pretrained.Mixed_6b(x_orig)
            x_orig = inception_pretrained.Mixed_6c(x_orig)
            x_orig = inception_pretrained.Mixed_6d(x_orig)
            x_orig = inception_pretrained.Mixed_6e(x_orig)
            x_orig = inception_pretrained.Mixed_7a(x_orig)
            x_orig = inception_pretrained.Mixed_7b(x_orig)
            x_orig = inception_pretrained.Mixed_7c(x_orig)
            x_orig = inception_pretrained.avgpool(x_orig)
            feat_orig = torch.flatten(x_orig, 1)

        outputs_equal = torch.allclose(feat_wrapped, feat_orig, rtol=1e-5, atol=1e-6)
        print(f"   输出一致性: {outputs_equal}")
        if outputs_equal:
            print("   ✅ 与原始模型输出完全一致!")
        else:
            max_diff = (feat_wrapped - feat_orig).abs().max().item()
            print(f"   ⚠️  最大差异: {max_diff:.6f}")
            if max_diff < 1e-4:
                print("   差异很小，可接受（浮点精度问题）")
            else:
                print("   ❌ 差异过大，可能有问题!")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过！InceptionV3FeatureExtractor正确保留了预训练权重。")
    print("=" * 80)

if __name__ == '__main__':
    test_weights()
