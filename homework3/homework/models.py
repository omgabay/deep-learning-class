from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.conv_layers = nn.Sequential(
            # Block 1: (b, 3, 64, 64) -> (b, 32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (b, 32, 32, 32) -> (b, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (b, 64, 16, 16) -> (b, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Global Average Pooling: (b, 128, 8, 8) -> (b, 128, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),          # (b, 128)
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        z = self.conv_layers(z)
        logits = self.classifier(z)

        return logits
    
    
    
    # class CNNBlock(torch.nn.Module):
    #     def __init__(self, in_channels, out_channels, stride):
    #         super().__init__()
    #         kernel_size = 3
    #         padding = (kernel_size - 1) // 2

    #         self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    #         self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
    #         self.c3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
    #         self.relu = torch.nn.ReLU()

    #     def forward(self, x):
    #         x = self.relu(self.c1(x))
    #         x = self.relu(self.c2(x))
    #         x = self.relu(self.c3(x))
    #         return x
        
    # def __init__(
    #     self,
    #     in_channels: int = 3,
    #     num_classes: int = 6,
    #     channels_l0: int = 24,
    #     num_cnn_blocks: int = 4,
    # ):
    #     """
    #     A convolutional network for image classification.

    #     Args:
    #         in_channels: int, number of input channels
    #         num_classes: int
    #     """
    #     super().__init__()

    #     self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
    #     self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

    #     out_channels = channels_l0
    #     cnn_layers = [
    #         torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5),
    #         torch.nn.ReLU(),
    #     ]
        
    #     in_channels = out_channels
    #     for _ in range(num_cnn_blocks):
    #         out_channels = 2 * in_channels
    #         cnn_layers.append(self.CNNBlock(in_channels, out_channels, stride=2))
    #         in_channels = out_channels
            
    #     cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
    #     cnn_layers.append(torch.nn.Flatten())
    #     cnn_layers.append(torch.nn.Linear(in_channels, num_classes))
    #     self.network = torch.nn.Sequential(*cnn_layers)
        
   

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         x: tensor (b, 3, h, w) image

    #     Returns:
    #         tensor (b, num_classes) logits
    #     """
    #     # optional: normalizes the input
    #     z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
    #     return self.network(z)
        

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        conv1_channel = 32
        conv2_channel = 64
        conv3_channel = 128
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv1_channel),
            nn.ReLU(),
            nn.Conv2d(conv1_channel, conv1_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv1_channel),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(conv1_channel, conv2_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv2_channel),
            nn.ReLU(),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(conv2_channel, conv3_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv3_channel),
            nn.ReLU(),
        )
        
        
        self.up2 = nn.ConvTranspose2d(conv3_channel, conv2_channel, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(conv2_channel * 2, conv2_channel, 3, padding=1),
            nn.BatchNorm2d(conv2_channel),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(conv2_channel, conv1_channel, 3, stride=2, padding=1, output_padding=1)

        self.dec1 = nn.Sequential(
            nn.Conv2d(conv1_channel * 2, conv1_channel, 3, padding=1),
            nn.BatchNorm2d(conv1_channel),
            nn.ReLU(),
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(conv1_channel, conv1_channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(conv1_channel),
            nn.ReLU(),
        )

        self.segmentation_head = nn.Conv2d(conv1_channel, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(conv1_channel, 1, kernel_size=1)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        e1 = self.enc1(z)   # (b, 32, 48, 64)
        e2 = self.enc2(e1)  # (b, 64, 24, 32)
        e3 = self.enc3(e2)  # (b, 128, 12, 16)

        # Decoder + skips
        d2 = self.up2(e3)               # (b, 64, 24, 32)
        d2 = torch.cat([d2, e2], dim=1)  # (b, 128, 24, 32)
        d2 = self.dec2(d2)               # (b, 64, 24, 32)

        d1 = self.up1(d2)               # (b, 32, 48, 64)
        d1 = torch.cat([d1, e1], dim=1)  # (b, 64, 48, 64)
        d1 = self.dec1(d1)               # (b, 32, 48, 64)

        features = self.up0(d1)         # (b, 32, 96, 128)
        
        # propagte features to segmentation to classify - left or right track pixel / background pixel.
        logits = self.segmentation_head(features)
        
        # propage to depth estimation network
        raw_depth = self.depth_head(features).squeeze(1)
                
        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = torch.sigmoid(raw_depth)

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)
    print(f"Classifier output shape: {output.shape}")
    print(f"Classifier size: {calculate_model_size_mb(model):.2f} MB")

    detector = load_model("detector", in_channels=3, num_classes=3).to(device)
    logits, raw_depth = detector(sample_batch)
    print(f"Detector logits shape: {logits.shape}")
    print(f"Detector depth shape: {raw_depth.shape}")
    print(f"Detector size: {calculate_model_size_mb(detector):.2f} MB")


if __name__ == "__main__":
    debug_model()
