"""Class activation mapping methods used by the inference pipeline."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GradCam:
    """Gradient-weighted class activation mapping."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.gradient_handle = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.feature_maps = output.detach()
        if output.requires_grad:
            self.gradient_handle = output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients: Tensor) -> None:
        self.gradients = gradients.detach()

    def __call__(self, class_scores: Tensor, predicted_class: int) -> Tensor:
        if self.feature_maps is None:
            raise RuntimeError("Grad-CAM target layer did not produce feature maps.")

        one_hot = torch.zeros_like(class_scores)
        one_hot[0, predicted_class] = 1
        self.model.zero_grad()
        class_scores.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("Grad-CAM target layer did not produce gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        return F.relu(cam)

    def close(self) -> None:
        self.forward_handle.remove()
        if self.gradient_handle is not None:
            self.gradient_handle.remove()


class ClCam:
    """Gradient-free CL-CAM using deterministic K-means.

    Feature channels are clustered by their spatial activation patterns, each
    cluster is represented by its channel mean, and target-class scores from
    masked forward passes select the cognition-base and cognition-scissors
    maps. Beta balances the two maps and defaults to 0.5.

    Representative maps are min-max normalized before masking so that the
    MobileViT activation scale cannot arbitrarily rescale the normalized input;
    this stability step is an explicit implementation choice.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        num_clusters: int = 6,
        beta: float = 0.5,
        seed: int = 0,
        max_iterations: int = 100,
    ) -> None:
        if num_clusters < 2:
            raise ValueError("CL-CAM requires at least two clusters.")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("CL-CAM beta must be between 0 and 1.")
        if max_iterations < 1:
            raise ValueError("CL-CAM max_iterations must be positive.")

        self.model = model
        self.num_clusters = num_clusters
        self.beta = beta
        self.seed = seed
        self.max_iterations = max_iterations
        self.feature_maps = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError(
                "CL-CAM target layer output must be a four-dimensional tensor."
            )
        self.feature_maps = output.detach()

    @staticmethod
    def _normalize_masks(masks: Tensor) -> Tensor:
        flat = masks.flatten(start_dim=1)
        minimum = flat.min(dim=1).values[:, None, None]
        maximum = flat.max(dim=1).values[:, None, None]
        return (masks - minimum) / (maximum - minimum).clamp_min(1e-8)

    def _initial_centers(self, features: Tensor) -> Tensor:
        """Initialize centers with a seeded K-means++ procedure on CPU."""
        num_channels = features.shape[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        first = int(torch.randint(num_channels, (1,), generator=generator).item())
        selected = [first]
        centers = [features[first]]

        while len(centers) < self.num_clusters:
            distances = torch.cdist(features, torch.stack(centers)).square()
            closest = distances.min(dim=1).values
            closest[selected] = 0
            total = closest.sum()
            if total <= 1e-12:
                next_index = next(
                    index for index in range(num_channels) if index not in selected
                )
            else:
                next_index = int(
                    torch.multinomial(
                        closest / total,
                        num_samples=1,
                        generator=generator,
                    ).item()
                )
            selected.append(next_index)
            centers.append(features[next_index])

        return torch.stack(centers)

    def _kmeans(self, features: Tensor) -> Tensor:
        """Cluster flattened channels and return one label per channel."""
        features = features.detach().to(device="cpu", dtype=torch.float32)
        num_channels = features.shape[0]
        if self.num_clusters > num_channels:
            raise ValueError(
                "CL-CAM cluster count cannot exceed the number of feature channels "
                f"({self.num_clusters} > {num_channels})."
            )

        centers = self._initial_centers(features)
        previous_labels = None

        for _ in range(self.max_iterations):
            distances = torch.cdist(features, centers)
            labels = distances.argmin(dim=1)

            # Identical or nearly identical centers can create an empty group.
            # Move a poorly represented channel from a group with at least two
            # members so that every representative remains well-defined.
            counts = torch.bincount(labels, minlength=self.num_clusters)
            nearest = distances.gather(1, labels[:, None]).squeeze(1)
            for group in range(self.num_clusters):
                if counts[group] > 0:
                    continue
                candidates = torch.argsort(nearest, descending=True)
                candidate = next(
                    int(index.item())
                    for index in candidates
                    if counts[labels[index]] > 1
                )
                donor_group = int(labels[candidate].item())
                labels[candidate] = group
                counts[donor_group] -= 1
                counts[group] += 1

            new_centers = torch.stack(
                [features[labels == group].mean(dim=0) for group in range(self.num_clusters)]
            )
            if previous_labels is not None and torch.equal(labels, previous_labels):
                centers = new_centers
                break
            previous_labels = labels.clone()
            centers = new_centers

        return labels

    def __call__(
        self,
        input_tensor: Tensor,
        predicted_class: int,
    ) -> Tuple[Tensor, Dict[str, object]]:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("CL-CAM currently expects one input image at a time.")
        if self.feature_maps is None:
            raise RuntimeError(
                "Run the model once after creating ClCam so the target-layer "
                "feature maps are available."
            )
        if self.feature_maps.shape[0] != 1:
            raise ValueError("CL-CAM target features must have batch size one.")

        feature_maps = self.feature_maps[0].detach()
        channels, feature_height, feature_width = feature_maps.shape
        labels_cpu = self._kmeans(feature_maps.reshape(channels, -1))
        labels = labels_cpu.to(feature_maps.device)

        representatives = torch.stack(
            [
                feature_maps[labels == group].mean(dim=0)
                for group in range(self.num_clusters)
            ]
        )
        masks = F.interpolate(
            representatives.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        masks = self._normalize_masks(masks)
        masked_inputs = input_tensor * masks.unsqueeze(1)

        with torch.inference_mode():
            masked_logits = self.model(masked_inputs)
            group_scores = torch.softmax(masked_logits, dim=1)[:, predicted_class]

        base_group = int(group_scores.argmax().item())
        scissors_group = int(group_scores.argmin().item())
        cam = (
            self.beta * representatives[base_group]
            - (1.0 - self.beta) * representatives[scissors_group]
        )
        cam = F.relu(cam).unsqueeze(0).unsqueeze(0)

        metadata: Dict[str, object] = {
            "num_clusters": self.num_clusters,
            "beta": self.beta,
            "seed": self.seed,
            "max_iterations": self.max_iterations,
            "base_group": base_group,
            "scissors_group": scissors_group,
            "group_scores": [round(float(score), 6) for score in group_scores.cpu()],
            "group_sizes": [
                int((labels_cpu == group).sum().item())
                for group in range(self.num_clusters)
            ],
            "masked_samples": self.num_clusters,
            "masked_forward_batches": 1,
            "target_feature_size": [feature_height, feature_width],
        }
        return cam, metadata

    def close(self) -> None:
        self.forward_handle.remove()
