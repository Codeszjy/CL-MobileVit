import unittest

import torch
from torch import nn

from cam_methods import ClCam, GradCam


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, 3)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


class CamMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyClassifier().eval()
        self.image = torch.randn(1, 3, 16, 16)

    def test_grad_cam_returns_finite_spatial_map(self):
        extractor = GradCam(self.model, self.model.features)
        try:
            logits = self.model(self.image)
            target = int(logits.argmax(dim=1).item())
            cam = extractor(logits, target)
        finally:
            extractor.close()

        self.assertEqual(tuple(cam.shape), (1, 1, 16, 16))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertGreaterEqual(float(cam.min()), 0.0)

    def test_cl_cam_is_gradient_free_and_preserves_predictions(self):
        extractor = ClCam(
            self.model,
            self.model.features,
            num_clusters=4,
            beta=0.5,
            seed=3,
        )
        try:
            with torch.inference_mode():
                logits_before = self.model(self.image)
            target = int(logits_before.argmax(dim=1).item())
            cam, metadata = extractor(self.image, target)
            with torch.inference_mode():
                logits_after = self.model(self.image)
        finally:
            extractor.close()

        self.assertEqual(tuple(cam.shape), (1, 1, 16, 16))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertTrue(torch.equal(logits_before, logits_after))
        self.assertEqual(metadata["num_clusters"], 4)
        self.assertEqual(metadata["max_iterations"], 100)
        self.assertEqual(metadata["masked_samples"], 4)
        self.assertEqual(metadata["masked_forward_batches"], 1)
        self.assertEqual(sum(metadata["group_sizes"]), 8)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_cl_cam_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ClCam(self.model, self.model.features, num_clusters=1)
        with self.assertRaises(ValueError):
            ClCam(self.model, self.model.features, beta=1.1)


if __name__ == "__main__":
    unittest.main()
