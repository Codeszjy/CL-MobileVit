import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from cam_methods import ClCam, GradCam
from medication_knowledge import (
    MODEL_DISEASE_RULES,
    SAFETY_NOTICES,
    MedicationKnowledgeBase,
    format_text_report,
    save_report,
)
from model import mobile_vit_small, mobile_vit_x_small, mobile_vit_xx_small


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FACTORIES = {
    "xx_small": mobile_vit_xx_small,
    "x_small": mobile_vit_x_small,
    "small": mobile_vit_small,
}
# =============================================================================
# Code-side inference configuration
# Edit this block when you prefer `python predict.py` without command-line
# arguments. Every value can still be overridden temporarily from the CLI.
# =============================================================================
DEFAULT_CONFIG = {
    "image": PROJECT_DIR / "test_images" / "rust_01.jpg",
    "weights": PROJECT_DIR / "weights" / "best_model.pth",
    "output_dir": PROJECT_DIR / "outputs" / "cl_cam",
    "model_variant": "auto",
    "class_indices": PROJECT_DIR / "class_indices.json",
    "pesticide_csv": (
        PROJECT_DIR / "data" / "knowledge" / "China_pesticides_wheat.csv"
    ),
    "confidence_threshold": 0.80,
    "top_k": 3,
    "allow_generic_rust": False,
    "device": "auto",
    "img_size": 224,

    "cam_method": "cl-cam",
    "cam_target_layer": "layer_5",
    "cam_clusters": 6,
    "cam_beta": 0.5,
    "cam_seed": 0,
    "cam_max_iterations": 100,
    "cam_overlay_alpha": 0.4,
}


def jet_colormap(values):
    """Map normalized values to an RGB JET-style heatmap using NumPy only."""
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.uint8(255 * np.stack((red, green, blue), axis=-1))


def show_cam_on_image(img, cam, heatmap_alpha):
    if not 0.0 <= heatmap_alpha <= 1.0:
        raise ValueError("CAM overlay alpha must be between 0 and 1.")

    # Upsample CAM to image size
    cam_array = cam.detach().cpu().numpy().squeeze().astype(np.float32)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    heatmap = np.asarray(
        Image.fromarray(cam_array).resize(
            (img.shape[1], img.shape[0]), resample=resampling
        ),
        dtype=np.float32,
    )
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap_colored = jet_colormap(heatmap)

    # Superimpose heatmap
    superimposed_img = np.uint8(
        heatmap_colored * heatmap_alpha + img * (1.0 - heatmap_alpha)
    )

    return superimposed_img, heatmap_colored


def save_cam_figure(
    original,
    heatmap,
    overlay,
    result_text,
    output_path,
    cam_title,
):
    """Save a publication-friendly three-panel image without Matplotlib."""
    panel_size = 448
    margin = 24
    title_height = 42
    canvas_width = panel_size * 3 + margin * 4
    canvas_height = panel_size + title_height + margin * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    panels = (
        (original.convert("RGB"), "Original Image"),
        (Image.fromarray(heatmap), f"{cam_title} Heatmap"),
        (Image.fromarray(overlay), f"Overlay: {result_text}"),
    )

    for index, (panel, title) in enumerate(panels):
        x = margin + index * (panel_size + margin)
        y = margin + title_height
        panel = panel.resize((panel_size, panel_size), resample=resampling)
        canvas.paste(panel, (x, y))
        text_box = draw.textbbox((0, 0), title)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x + (panel_size - text_width) / 2, margin + 10),
            title,
            fill="black",
        )

    canvas.save(output_path, format="PNG", dpi=(300, 300))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "CL-MobileViT wheat disease recognition, CAM explanation, "
            "and safety-gated pesticide knowledge retrieval"
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_CONFIG["image"],
        help="待识别的小麦叶片图像",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_CONFIG["weights"],
        help="CL-MobileViT模型权重",
    )
    parser.add_argument(
        "--model-variant",
        choices=("auto", "xx_small", "x_small", "small"),
        default=DEFAULT_CONFIG["model_variant"],
        help="模型规格；auto会根据权重通道数自动识别",
    )
    parser.add_argument(
        "--class-indices",
        type=Path,
        default=DEFAULT_CONFIG["class_indices"],
        help="类别索引JSON文件",
    )
    parser.add_argument(
        "--pesticide-csv",
        type=Path,
        default=DEFAULT_CONFIG["pesticide_csv"],
        help="小麦农药登记知识CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG["output_dir"],
        help="CAM与知识推理报告输出目录",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIG["confidence_threshold"],
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG["top_k"])
    parser.add_argument(
        "--cam-method",
        choices=("cl-cam", "grad-cam"),
        default=DEFAULT_CONFIG["cam_method"],
        help="可解释性方法；默认使用无梯度的 CL-CAM",
    )
    parser.add_argument(
        "--cam-target-layer",
        default=DEFAULT_CONFIG["cam_target_layer"],
        help="CAM目标层名称，例如layer_5或layer_4",
    )
    parser.add_argument(
        "--cam-clusters",
        type=int,
        default=DEFAULT_CONFIG["cam_clusters"],
        help="CL-CAM的特征通道聚类数Q",
    )
    parser.add_argument(
        "--cam-beta",
        type=float,
        default=DEFAULT_CONFIG["cam_beta"],
        help="CL-CAM中base与scissors特征图的平衡系数",
    )
    parser.add_argument(
        "--cam-seed",
        type=int,
        default=DEFAULT_CONFIG["cam_seed"],
        help="CL-CAM K-means初始化随机种子",
    )
    parser.add_argument(
        "--cam-max-iterations",
        type=int,
        default=DEFAULT_CONFIG["cam_max_iterations"],
        help="CL-CAM K-means最大迭代次数",
    )
    parser.add_argument(
        "--cam-overlay-alpha",
        type=float,
        default=DEFAULT_CONFIG["cam_overlay_alpha"],
        help="热力图在叠加图中的透明度，取值范围为0到1",
    )
    parser.add_argument(
        "--allow-generic-rust",
        action="store_true",
        default=DEFAULT_CONFIG["allow_generic_rust"],
        help="允许叶锈病额外检索‘锈病’泛化记录（需专家复核）",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_CONFIG["device"],
        help="推理设备，默认自动选择",
    )
    parser.add_argument(
        "--img-size", type=int, default=DEFAULT_CONFIG["img_size"]
    )
    return parser


def resolve_device(device_name):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已指定CUDA，但当前环境未检测到可用GPU。")
        return torch.device("cuda:0")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_target_layer(model, target_layer_name):
    """Resolve a dotted module path such as ``layer_5`` or ``layer_5.1``."""
    try:
        target_layer = model.get_submodule(target_layer_name)
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"CAM target layer does not exist: {target_layer_name}"
        ) from error
    return target_layer


def normalize_state_dict(state_dict):
    """Remove a possible DataParallel prefix without changing tensor values."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(weight_path, device):
    """Prefer PyTorch's safer weights-only loader when the version supports it."""
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def infer_model_variant(state_dict):
    classifier_weight = state_dict.get("classifier.fc.weight")
    if classifier_weight is None or classifier_weight.ndim != 2:
        raise RuntimeError(
            "无法从权重自动识别MobileViT规格，请使用--model-variant明确指定。"
        )

    feature_dimension = int(classifier_weight.shape[1])
    dimension_to_variant = {320: "xx_small", 384: "x_small", 640: "small"}
    try:
        return dimension_to_variant[feature_dimension]
    except KeyError as error:
        raise RuntimeError(
            f"无法识别分类器输入维度{feature_dimension}对应的MobileViT规格。"
        ) from error


def create_failed_closed_report(args, predicted_class, confidence, error):
    rule = MODEL_DISEASE_RULES.get(predicted_class, {})
    return {
        "model_name": "CL-MobileViT",
        "predicted_class": predicted_class,
        "disease_cn": rule.get("disease_cn", "未知类别"),
        "confidence": round(float(confidence), 6),
        "confidence_threshold": args.confidence_threshold,
        "status": "knowledge_base_unavailable_manual_review",
        "message": f"知识库读取失败，系统已拒绝自动推荐：{error}",
        "data_source": str(args.pesticide_csv.resolve()),
        "safety_notices": list(SAFETY_NOTICES),
        "candidates": [],
    }


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    img_size = args.img_size
    spatial_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size)]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    # Load image.
    img_path = args.image.expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {img_path}")
    
    # Use the exact resized and center-cropped pixels seen by the model for the
    # heatmap overlay. This avoids spatial misalignment on non-square images.
    pil_img = Image.open(img_path).convert("RGB")
    model_img = spatial_transform(pil_img)
    display_img = np.asarray(model_img, dtype=np.uint8)

    # [N, C, H, W]
    tensor_img = tensor_transform(model_img).unsqueeze(0).to(device)

    # read class_indict
    json_path = args.class_indices.expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Class index file does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        class_indict = json.load(f)

    model_weight_path = args.weights.expanduser().resolve()
    if not model_weight_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {model_weight_path}")
    checkpoint = load_checkpoint(model_weight_path, device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = normalize_state_dict(state_dict)

    model_variant = (
        infer_model_variant(state_dict)
        if args.model_variant == "auto"
        else args.model_variant
    )
    model = MODEL_FACTORIES[model_variant](num_classes=len(class_indict)).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            f"权重与模型规格'{model_variant}'不匹配。"
            "请使用--model-variant auto或选择正确的规格。"
        ) from error
    model.eval()
    print(f"model variant: {model_variant}   device: {device}")

    # Prediction and the selected CAM method share the first forward result.
    target_layer = resolve_target_layer(model, args.cam_target_layer)
    cam_metadata = {
        "target_layer": args.cam_target_layer,
        "overlay_alpha": args.cam_overlay_alpha,
    }
    is_cl_cam = args.cam_method == "cl-cam"

    if is_cl_cam:
        cam_extractor = ClCam(
            model=model,
            target_layer=target_layer,
            num_clusters=args.cam_clusters,
            beta=args.cam_beta,
            seed=args.cam_seed,
            max_iterations=args.cam_max_iterations,
        )
        with torch.inference_mode():
            output = model(tensor_img)
    else:
        cam_extractor = GradCam(model=model, target_layer=target_layer)
        output = model(tensor_img)

    predict = torch.softmax(output.detach(), dim=1).squeeze(0).cpu()
    predict_cla = int(torch.argmax(predict).item())
    try:
        if is_cl_cam:
            cam, cl_cam_metadata = cam_extractor(tensor_img, predict_cla)
            cam_metadata.update(cl_cam_metadata)
        else:
            cam = cam_extractor(output, predict_cla)
            cam_metadata["masked_samples"] = 0
            cam_metadata["masked_forward_batches"] = 0
    finally:
        cam_extractor.close()
    cam_image, heatmap = show_cam_on_image(
        display_img,
        cam,
        heatmap_alpha=args.cam_overlay_alpha,
    )

    predicted_class = class_indict[str(predict_cla)]
    confidence = float(predict[predict_cla].item())
    class_probabilities = {
        class_indict[str(i)]: round(float(predict[i].item()), 6)
        for i in range(len(predict))
    }

    # Display classification results.
    print_res = "class: {}   prob: {:.3f}".format(predicted_class, confidence)
    print(print_res)
    for i in range(len(predict)):
        print(
            "class: {:15}   prob: {:.3f}".format(
                class_indict[str(i)], float(predict[i].item())
            )
        )
    
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{args.cam_method.replace('-', '_')}_result.png"
    cam_title = "CL-CAM" if is_cl_cam else "Grad-CAM"
    save_cam_figure(
        original=model_img,
        heatmap=heatmap,
        overlay=cam_image,
        result_text=print_res,
        output_path=output_filename,
        cam_title=cam_title,
    )
    print(f"\n{cam_title} visualization saved to: {output_filename}")

    # Safety-gated pesticide knowledge retrieval. Any knowledge-layer failure
    # fails closed: classification remains available, but no candidate is shown.
    try:
        knowledge_base = MedicationKnowledgeBase(
            csv_path=args.pesticide_csv,
            confidence_threshold=args.confidence_threshold,
            allow_generic_rust=args.allow_generic_rust,
        )
        report = knowledge_base.lookup(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=args.top_k,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        report = create_failed_closed_report(
            args=args,
            predicted_class=predicted_class,
            confidence=confidence,
            error=error,
        )

    report.update(
        {
            "model_name": "CL-MobileViT",
            "model_variant": model_variant,
            "image_path": str(img_path),
            "model_weights": str(model_weight_path),
            "cam_method": args.cam_method,
            "cam_path": str(output_filename),
            "cam_parameters": cam_metadata,
            "class_probabilities": class_probabilities,
        }
    )
    recommendation_json, recommendation_text = save_report(report, output_dir)

    print("\n" + format_text_report(report))
    print(f"Medication JSON report saved to: {recommendation_json}")
    print(f"Medication text report saved to: {recommendation_text}")


if __name__ == '__main__':
    main()
