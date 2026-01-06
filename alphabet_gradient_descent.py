# ================================================================
# ALPHABET GRADIENT DESCENT TRAINER (Standalone)
# ================================================================
# Same task as evolutionary alphabet trainer, but using backpropagation.
#
# Architecture: 10000 pixels -> 32 hidden -> 26 outputs (letters)
# This is intentionally undersized to see how gradient descent handles
# a constrained capacity network on this visual recognition task.
#
# Extensive logging to track:
# - Loss curves (train/validation)
# - Per-letter accuracy breakdown
# - Gradient statistics (mean, std, max, min)
# - Weight statistics
# - Learning rate scheduling
# - Confusion patterns
# ================================================================

import os
import sys
import time
import platform
import random
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Alphabet Recognition with Gradient Descent")
parser.add_argument("--fonts", type=int, default=5, choices=[1, 2, 3, 4, 5],
                    help="Number of fonts to use for training (1-5, default: 5)")
args = parser.parse_args()


# ================================================================
# CONFIGURATION
# ================================================================
class Config:
    # Image dimensions
    FIELD_WIDTH = 100
    FIELD_HEIGHT = 100
    INPUT_SIZE = FIELD_WIDTH * FIELD_HEIGHT  # 10000

    # Network architecture (intentionally small!)
    HIDDEN_SIZE = 32  # Very constrained - will this work?
    OUTPUT_SIZE = 26  # a-z

    # Training parameters
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    EPOCHS = 10000  # High limit - early stopping will handle termination

    # Early stopping
    PATIENCE = 300  # Stop after N epochs without improvement
    MIN_IMPROVEMENT = 0.1  # Minimum accuracy improvement to count as progress
    DRAMATIC_DROP = 5.0  # Stop if accuracy drops by this much from best

    # Data generation
    VARIATIONS_PER_LETTER = 50  # Training variations per letter
    VAL_VARIATIONS_PER_LETTER = 10  # Validation variations per letter

    # Augmentation
    MAX_ROTATION = 25  # degrees
    MAX_JITTER_RATIO = 0.2  # fraction of image size
    FONT_SIZE = 64

    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    DETAILED_LOG_INTERVAL = 1  # Detailed stats every N epochs

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# FONT LOADER (Cross-platform)
# ================================================================
class FontLoader:
    """Cross-platform font loading with detailed logging."""

    def __init__(self, max_fonts=5):
        self.fonts = []
        self.max_fonts = max_fonts
        self._load_fonts()

    def _load_fonts(self):
        """Load fonts based on detected platform."""
        current_platform = platform.system().lower()
        print(f"\n{'='*60}")
        print("FONT LOADING")
        print(f"{'='*60}")
        print(f"[FONTS] Detected platform: {platform.system()}")
        print(f"[FONTS] Max fonts to load: {self.max_fonts}")

        small_size = max(8, int(Config.FONT_SIZE * 12 / 64))
        normal_size = Config.FONT_SIZE

        # Platform-specific font candidates
        if current_platform == "windows":
            font_candidates = [
                ("C:\\Windows\\Fonts\\arial.ttf", "Arial"),
                ("C:\\Windows\\Fonts\\calibri.ttf", "Calibri"),
                ("C:\\Windows\\Fonts\\verdana.ttf", "Verdana"),
                ("C:\\Windows\\Fonts\\tahoma.ttf", "Tahoma"),
                ("C:\\Windows\\Fonts\\segoeui.ttf", "SegoeUI"),
                ("C:\\Windows\\Fonts\\times.ttf", "TimesNewRoman"),
                ("C:\\Windows\\Fonts\\georgia.ttf", "Georgia"),
                ("C:\\Windows\\Fonts\\consola.ttf", "Consolas"),
                ("C:\\Windows\\Fonts\\cour.ttf", "CourierNew"),
                ("C:\\Windows\\Fonts\\arialbd.ttf", "ArialBold"),
            ]
        elif current_platform == "darwin":
            font_candidates = [
                ("/System/Library/Fonts/Helvetica.ttc", "Helvetica"),
                ("/Library/Fonts/Arial.ttf", "Arial"),
                ("/System/Library/Fonts/Times.ttc", "Times"),
                ("/System/Library/Fonts/Monaco.ttf", "Monaco"),
                ("/System/Library/Fonts/Menlo.ttc", "Menlo"),
            ]
        else:
            # Linux
            font_candidates = [
                ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans"),
                ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "LiberationSans"),
                ("/usr/share/fonts/truetype/freefont/FreeSans.ttf", "FreeSans"),
                ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "DejaVuSerif"),
                ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuMono"),
            ]

        print(f"[FONTS] Attempting to load fonts...")

        for font_path, font_name in font_candidates:
            if len(self.fonts) >= self.max_fonts:
                break
            if os.path.exists(font_path):
                try:
                    font_small = ImageFont.truetype(font_path, small_size)
                    font_normal = ImageFont.truetype(font_path, normal_size)
                    self.fonts.append({
                        'name': font_name,
                        'path': font_path,
                        'small': font_small,
                        'normal': font_normal,
                    })
                    print(f"[FONTS]   OK  Loaded: {font_name}")
                except Exception as e:
                    print(f"[FONTS]   ERR Failed: {font_name} - {e}")

        # Fallback to default
        if not self.fonts:
            default_font = ImageFont.load_default()
            self.fonts.append({
                'name': 'Default',
                'path': None,
                'small': default_font,
                'normal': default_font,
            })
            print(f"[FONTS]   WARN Using PIL default font (no system fonts found)")

        font_names = [f['name'] for f in self.fonts]
        print(f"[FONTS] SUCCESS: Loaded {len(self.fonts)}/{self.max_fonts} fonts: {', '.join(font_names)}")
        print(f"{'='*60}\n")

    def get_font(self, idx, size='normal'):
        """Get a font by index."""
        font_data = self.fonts[idx % len(self.fonts)]
        return font_data[size]

    def num_fonts(self):
        return len(self.fonts)


# ================================================================
# LETTER RENDERER
# ================================================================
class LetterRenderer:
    """Renders letters with augmentation (rotation, jitter, font variation)."""

    def __init__(self, font_loader):
        self.font_loader = font_loader
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.width = Config.FIELD_WIDTH
        self.height = Config.FIELD_HEIGHT

    def render(self, letter_idx, variation_idx):
        """
        Render a letter with augmentation.

        Returns: numpy array of shape (H*W,) normalized to [0, 1]
        """
        letter = self.letters[letter_idx]

        # Determine augmentation from variation_idx
        num_fonts = self.font_loader.num_fonts()
        font_idx = variation_idx % num_fonts
        style_idx = (variation_idx // num_fonts) % 4

        use_small_font = style_idx < 2
        use_white_on_black = (style_idx % 2) == 0

        # Get font
        font = self.font_loader.get_font(font_idx, 'small' if use_small_font else 'normal')

        # Colors
        if use_white_on_black:
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
        else:
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)

        # Create image
        img = Image.new('RGB', (self.width, self.height), bg_color)

        # Create canvas for rotation
        padding = int(max(self.width, self.height) * 0.5)
        canvas_size = max(self.width, self.height) + 2 * padding
        letter_canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        letter_draw = ImageDraw.Draw(letter_canvas)

        # Get text bounds
        bbox = letter_draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw centered
        text_x = (canvas_size - text_width) // 2 - bbox[0]
        text_y = (canvas_size - text_height) // 2 - bbox[1]
        letter_draw.text((text_x, text_y), letter, font=font, fill=text_color + (255,))

        # Random rotation and jitter (seeded by variation_idx for reproducibility)
        random.seed(variation_idx * 1000 + letter_idx)
        rotation = random.uniform(-Config.MAX_ROTATION, Config.MAX_ROTATION)
        rotated = letter_canvas.rotate(rotation, resample=Image.BICUBIC, expand=False)

        # Jitter
        max_jitter_x = int(self.width * Config.MAX_JITTER_RATIO)
        max_jitter_y = int(self.height * Config.MAX_JITTER_RATIO)
        jitter_x = random.randint(-max_jitter_x, max_jitter_x)
        jitter_y = random.randint(-max_jitter_y, max_jitter_y)

        paste_x = (self.width - canvas_size) // 2 + jitter_x
        paste_y = (self.height - canvas_size) // 2 + jitter_y

        img.paste(rotated, (paste_x, paste_y), rotated)

        # Convert to grayscale numpy
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32) / 255.0
        return arr.flatten()


# ================================================================
# DATASET
# ================================================================
class AlphabetDataset(Dataset):
    """PyTorch dataset for alphabet recognition."""

    def __init__(self, renderer, variations_per_letter, is_training=True):
        self.renderer = renderer
        self.variations_per_letter = variations_per_letter
        self.is_training = is_training

        # Pre-generate all images
        print(f"[DATASET] Generating {'training' if is_training else 'validation'} data...")
        print(f"[DATASET] 26 letters x {variations_per_letter} variations = {26 * variations_per_letter} images")

        self.images = []
        self.labels = []

        start_time = time.time()
        for letter_idx in range(26):
            for var_idx in range(variations_per_letter):
                # Offset variation index for validation to get different images
                actual_var_idx = var_idx if is_training else var_idx + 10000
                img = renderer.render(letter_idx, actual_var_idx)
                self.images.append(img)
                self.labels.append(letter_idx)

        self.images = np.stack(self.images, axis=0)
        self.labels = np.array(self.labels)

        elapsed = time.time() - start_time
        print(f"[DATASET] Generated {len(self.images)} images in {elapsed:.2f}s")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.images[idx]).float(),
            torch.tensor(self.labels[idx]).long()
        )


# ================================================================
# NEURAL NETWORK
# ================================================================
class AlphabetNet(nn.Module):
    """
    Simple 2-layer network for alphabet recognition.

    Architecture: Input(10000) -> Hidden(32) -> Output(26)

    This is intentionally undersized to test gradient descent
    with limited capacity.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(Config.INPUT_SIZE, Config.HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(Config.HIDDEN_SIZE, Config.OUTPUT_SIZE)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        print(f"\n{'='*60}")
        print("NETWORK ARCHITECTURE")
        print(f"{'='*60}")
        print(f"  Input:  {Config.INPUT_SIZE} neurons (100x100 pixels)")
        print(f"  Hidden: {Config.HIDDEN_SIZE} neurons (ReLU activation)")
        print(f"  Output: {Config.OUTPUT_SIZE} neurons (26 letters)")
        print(f"")
        print(f"  Parameters:")
        print(f"    fc1.weight: {Config.INPUT_SIZE} x {Config.HIDDEN_SIZE} = {Config.INPUT_SIZE * Config.HIDDEN_SIZE:,}")
        print(f"    fc1.bias:   {Config.HIDDEN_SIZE}")
        print(f"    fc2.weight: {Config.HIDDEN_SIZE} x {Config.OUTPUT_SIZE} = {Config.HIDDEN_SIZE * Config.OUTPUT_SIZE:,}")
        print(f"    fc2.bias:   {Config.OUTPUT_SIZE}")
        total_params = (Config.INPUT_SIZE * Config.HIDDEN_SIZE + Config.HIDDEN_SIZE +
                       Config.HIDDEN_SIZE * Config.OUTPUT_SIZE + Config.OUTPUT_SIZE)
        print(f"    TOTAL:      {total_params:,} parameters")
        print(f"")
        print(f"  Bottleneck Analysis:")
        print(f"    Information compression: {Config.INPUT_SIZE} -> {Config.HIDDEN_SIZE}")
        print(f"    Compression ratio: {Config.INPUT_SIZE / Config.HIDDEN_SIZE:.1f}x")
        print(f"    Bits per letter (theoretical): {np.log2(26):.2f}")
        print(f"    Hidden units per letter: {Config.HIDDEN_SIZE / 26:.2f}")
        print(f"{'='*60}\n")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ================================================================
# TRAINING LOGGER
# ================================================================
class TrainingLogger:
    """Extensive logging for gradient descent training."""

    def __init__(self):
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.gradient_stats = []
        self.weight_stats = []
        self.per_letter_accuracy = defaultdict(list)
        self.confusion_matrix = np.zeros((26, 26), dtype=int)
        self.start_time = time.time()
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # JSON log data
        self.json_log = {
            "model_type": "gradient_descent",
            "architecture": {
                "input_size": Config.INPUT_SIZE,
                "hidden_size": Config.HIDDEN_SIZE,
                "output_size": Config.OUTPUT_SIZE,
                "total_params": (Config.INPUT_SIZE * Config.HIDDEN_SIZE + Config.HIDDEN_SIZE +
                                Config.HIDDEN_SIZE * Config.OUTPUT_SIZE + Config.OUTPUT_SIZE),
                "compression_ratio": Config.INPUT_SIZE / Config.HIDDEN_SIZE,
            },
            "training_config": {
                "learning_rate": Config.LEARNING_RATE,
                "batch_size": Config.BATCH_SIZE,
                "patience": Config.PATIENCE,
                "min_improvement": Config.MIN_IMPROVEMENT,
                "dramatic_drop_threshold": Config.DRAMATIC_DROP,
                "variations_per_letter_train": Config.VARIATIONS_PER_LETTER,
                "variations_per_letter_val": Config.VAL_VARIATIONS_PER_LETTER,
                "num_fonts": args.fonts,
            },
            "augmentation": {
                "description": "Each variation has unique randomized augmentation seeded by (variation_idx * 1000 + letter_idx)",
                "train_val_separation": "Training uses variation indices 0-N, Validation uses indices 10000-10000+N (completely different random seeds)",
                "rotation": {
                    "enabled": True,
                    "range_degrees": [-Config.MAX_ROTATION, Config.MAX_ROTATION],
                },
                "position_jitter": {
                    "enabled": True,
                    "max_ratio": Config.MAX_JITTER_RATIO,
                    "max_pixels": int(Config.FIELD_WIDTH * Config.MAX_JITTER_RATIO),
                },
                "font_variation": {
                    "enabled": True,
                    "num_fonts_used": args.fonts,
                    "font_cycling": "variation_idx % num_fonts",
                },
                "size_variation": {
                    "enabled": True,
                    "sizes": ["small (12pt scaled)", "normal (64pt)"],
                    "selection": "based on (variation_idx // num_fonts) % 4",
                },
                "color_inversion": {
                    "enabled": True,
                    "schemes": ["white_on_black", "black_on_white"],
                    "selection": "based on (variation_idx // num_fonts) % 4",
                },
                "total_style_combinations": "num_fonts * 2 sizes * 2 colors = unique base styles per variation index",
                "randomization_note": "Rotation and jitter are randomly sampled per variation using deterministic seed, ensuring train/val never overlap",
            },
            "platform": platform.system(),
            "device": Config.DEVICE,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "best": {
                "accuracy": 0.0,
                "epoch": 0,
                "loss": float('inf'),
            },
            "final": {},
            "stop_reason": None,
            "per_letter_final": {},
            "confusion_top10": [],
        }

    def log_batch(self, epoch, batch_idx, total_batches, loss, correct, total):
        """Log batch-level statistics."""
        acc = correct / total * 100
        print(f"  Batch {batch_idx+1:>3}/{total_batches} | Loss: {loss:.4f} | Acc: {acc:>5.1f}% ({correct}/{total})")

    def log_gradients(self, model, epoch):
        """Log gradient statistics."""
        grad_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'norm': grad.norm().item(),
                }

        self.gradient_stats.append(grad_stats)
        return grad_stats

    def log_weights(self, model, epoch):
        """Log weight statistics."""
        weight_stats = {}

        for name, param in model.named_parameters():
            data = param.data
            weight_stats[name] = {
                'mean': data.mean().item(),
                'std': data.std().item(),
                'min': data.min().item(),
                'max': data.max().item(),
                'norm': data.norm().item(),
            }

        self.weight_stats.append(weight_stats)
        return weight_stats

    def log_epoch_summary(self, epoch, train_loss, train_acc, val_loss, val_acc,
                          grad_stats, weight_stats, per_letter_acc, lr):
        """Log comprehensive epoch summary."""
        self.epoch_losses.append(train_loss)
        self.epoch_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        elapsed = time.time() - self.start_time

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*70}")
        print(f"  Time elapsed: {elapsed/60:.1f} min")
        print(f"  Learning rate: {lr:.6f}")
        print(f"")
        print(f"  Training:")
        print(f"    Loss:     {train_loss:.4f}")
        print(f"    Accuracy: {train_acc:.2f}%")
        print(f"")
        print(f"  Validation:")
        print(f"    Loss:     {val_loss:.4f}")
        print(f"    Accuracy: {val_acc:.2f}%")
        print(f"")

        # Gradient statistics
        print(f"  Gradient Statistics:")
        for name, stats in grad_stats.items():
            print(f"    {name}:")
            print(f"      mean={stats['mean']:>9.6f}  std={stats['std']:>9.6f}  norm={stats['norm']:>9.4f}")
            print(f"      min={stats['min']:>10.6f}  max={stats['max']:>9.6f}")

        print(f"")

        # Weight statistics
        print(f"  Weight Statistics:")
        for name, stats in weight_stats.items():
            print(f"    {name}:")
            print(f"      mean={stats['mean']:>9.6f}  std={stats['std']:>9.6f}  norm={stats['norm']:>9.4f}")

        print(f"")

        # Per-letter accuracy
        print(f"  Per-Letter Accuracy:")
        for letter_idx, acc in per_letter_acc.items():
            self.per_letter_accuracy[letter_idx].append(acc)

        # Display in two rows
        letters_row1 = self.letters[:13]
        letters_row2 = self.letters[13:]

        row1_str = "    "
        for i, letter in enumerate(letters_row1):
            acc = per_letter_acc.get(i, 0)
            if acc >= 90:
                row1_str += f"{letter}:OK  "
            elif acc >= 50:
                row1_str += f"{letter}:{acc:>2.0f}% "
            else:
                row1_str += f"{letter}:--  "
        print(row1_str)

        row2_str = "    "
        for i, letter in enumerate(letters_row2, start=13):
            acc = per_letter_acc.get(i, 0)
            if acc >= 90:
                row2_str += f"{letter}:OK  "
            elif acc >= 50:
                row2_str += f"{letter}:{acc:>2.0f}% "
            else:
                row2_str += f"{letter}:--  "
        print(row2_str)

        # Mastery count
        mastered = sum(1 for acc in per_letter_acc.values() if acc >= 90)
        struggling = sum(1 for acc in per_letter_acc.values() if acc < 50)
        print(f"")
        print(f"  Mastered (>=90%): {mastered}/26")
        print(f"  Struggling (<50%): {struggling}/26")

        # Loss trend
        if len(self.epoch_losses) > 1:
            loss_delta = train_loss - self.epoch_losses[-2]
            acc_delta = train_acc - self.epoch_accuracies[-2]
            print(f"")
            print(f"  Trends:")
            print(f"    Loss delta:     {loss_delta:+.4f} ({'improving' if loss_delta < 0 else 'worsening'})")
            print(f"    Accuracy delta: {acc_delta:+.2f}% ({'improving' if acc_delta > 0 else 'worsening'})")

        print(f"{'='*70}\n")

    def update_confusion(self, predictions, labels):
        """Update confusion matrix."""
        for pred, label in zip(predictions, labels):
            self.confusion_matrix[label, pred] += 1

    def print_confusion_summary(self):
        """Print confusion matrix summary (most confused pairs)."""
        print(f"\n{'='*70}")
        print("CONFUSION ANALYSIS")
        print(f"{'='*70}")

        # Find most confused pairs (excluding diagonal)
        confusion_pairs = []
        for true_idx in range(26):
            for pred_idx in range(26):
                if true_idx != pred_idx:
                    count = self.confusion_matrix[true_idx, pred_idx]
                    if count > 0:
                        confusion_pairs.append((
                            self.letters[true_idx],
                            self.letters[pred_idx],
                            count
                        ))

        confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        print("  Most Confused Pairs (True -> Predicted):")
        for true_letter, pred_letter, count in confusion_pairs[:15]:
            print(f"    {true_letter} -> {pred_letter}: {count} times")

        print(f"{'='*70}\n")

    def print_final_summary(self):
        """Print final training summary."""
        elapsed = time.time() - self.start_time

        print(f"\n{'='*70}")
        print("FINAL TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print(f"  Epochs: {len(self.epoch_losses)}")
        print(f"")
        print(f"  Best Training:")
        print(f"    Loss:     {min(self.epoch_losses):.4f} (epoch {np.argmin(self.epoch_losses)+1})")
        print(f"    Accuracy: {max(self.epoch_accuracies):.2f}% (epoch {np.argmax(self.epoch_accuracies)+1})")
        print(f"")
        print(f"  Best Validation:")
        print(f"    Loss:     {min(self.val_losses):.4f} (epoch {np.argmin(self.val_losses)+1})")
        print(f"    Accuracy: {max(self.val_accuracies):.2f}% (epoch {np.argmax(self.val_accuracies)+1})")
        print(f"")
        print(f"  Final Training:")
        print(f"    Loss:     {self.epoch_losses[-1]:.4f}")
        print(f"    Accuracy: {self.epoch_accuracies[-1]:.2f}%")
        print(f"")
        print(f"  Final Validation:")
        print(f"    Loss:     {self.val_losses[-1]:.4f}")
        print(f"    Accuracy: {self.val_accuracies[-1]:.2f}%")
        print(f"")

        # Random baseline comparison
        random_acc = 100 / 26
        print(f"  Comparison to Random Baseline ({random_acc:.2f}%):")
        improvement = self.val_accuracies[-1] - random_acc
        print(f"    Improvement: {improvement:+.2f}%")
        print(f"    Multiplier:  {self.val_accuracies[-1] / random_acc:.2f}x better than random")

        print(f"{'='*70}\n")

        self.print_confusion_summary()

    def log_epoch_json(self, epoch, train_loss, train_acc, val_loss, val_acc,
                       per_letter_acc, lr, grad_norms):
        """Log epoch data to JSON structure."""
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 2),
            "learning_rate": lr,
            "time_elapsed_sec": round(time.time() - self.start_time, 1),
            "grad_norm_fc1": round(grad_norms.get('fc1.weight', 0), 6),
            "grad_norm_fc2": round(grad_norms.get('fc2.weight', 0), 6),
            "letters_mastered": sum(1 for acc in per_letter_acc.values() if acc >= 90),
            "letters_struggling": sum(1 for acc in per_letter_acc.values() if acc < 50),
        }

        self.json_log["epochs"].append(epoch_data)

        # Update best
        if val_acc > self.json_log["best"]["accuracy"]:
            self.json_log["best"]["accuracy"] = round(val_acc, 2)
            self.json_log["best"]["epoch"] = epoch + 1
            self.json_log["best"]["loss"] = round(val_loss, 6)

    def finalize_json(self, stop_reason, final_val_acc, final_val_loss, final_per_letter):
        """Finalize the JSON log with summary data."""
        self.json_log["end_time"] = datetime.now().isoformat()
        self.json_log["total_time_sec"] = round(time.time() - self.start_time, 1)
        self.json_log["total_epochs"] = len(self.json_log["epochs"])
        self.json_log["stop_reason"] = stop_reason

        self.json_log["final"] = {
            "val_accuracy": round(final_val_acc, 2),
            "val_loss": round(final_val_loss, 6),
            "train_accuracy": round(self.epoch_accuracies[-1], 2) if self.epoch_accuracies else 0,
            "train_loss": round(self.epoch_losses[-1], 6) if self.epoch_losses else 0,
        }

        # Per-letter final accuracy
        for letter_idx, acc in final_per_letter.items():
            letter = self.letters[letter_idx]
            self.json_log["per_letter_final"][letter] = round(acc, 2)

        # Top confusion pairs
        confusion_pairs = []
        for true_idx in range(26):
            for pred_idx in range(26):
                if true_idx != pred_idx:
                    count = int(self.confusion_matrix[true_idx, pred_idx])
                    if count > 0:
                        confusion_pairs.append({
                            "true": self.letters[true_idx],
                            "predicted": self.letters[pred_idx],
                            "count": count
                        })
        confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
        self.json_log["confusion_top10"] = confusion_pairs[:10]

        # Summary stats
        random_baseline = 100 / 26
        self.json_log["summary"] = {
            "random_baseline": round(random_baseline, 2),
            "improvement_over_random": round(final_val_acc - random_baseline, 2),
            "multiplier_vs_random": round(final_val_acc / random_baseline, 2),
            "peak_accuracy": round(max(self.val_accuracies) if self.val_accuracies else 0, 2),
            "peak_epoch": int(np.argmax(self.val_accuracies) + 1) if self.val_accuracies else 0,
            "final_letters_mastered": sum(1 for acc in final_per_letter.values() if acc >= 90),
        }

    def save_json(self, filepath):
        """Save JSON log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.json_log, f, indent=2)
        print(f"[SAVE] Training log saved to: {filepath}")


# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def evaluate(model, dataloader, criterion, device, logger=None):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_letter_correct = defaultdict(int)
    per_letter_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Per-letter tracking
            for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                per_letter_total[label] += 1
                if pred == label:
                    per_letter_correct[label] += 1

            # Update confusion matrix
            if logger:
                logger.update_confusion(predicted.cpu().numpy(), labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    per_letter_acc = {}
    for letter_idx in range(26):
        if per_letter_total[letter_idx] > 0:
            per_letter_acc[letter_idx] = per_letter_correct[letter_idx] / per_letter_total[letter_idx] * 100
        else:
            per_letter_acc[letter_idx] = 0

    return avg_loss, accuracy, per_letter_acc


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(labels).sum().item()
        correct += batch_correct
        total += labels.size(0)

        # Log batch
        if (batch_idx + 1) % Config.LOG_INTERVAL == 0 or batch_idx == num_batches - 1:
            logger.log_batch(epoch, batch_idx, num_batches, loss.item(),
                           batch_correct, labels.size(0))

    return total_loss / total, correct / total * 100


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger):
    """Main training loop with early stopping."""
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Max Epochs: {Config.EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"")
    print(f"  Early Stopping:")
    print(f"    Patience: {Config.PATIENCE} epochs without improvement")
    print(f"    Min improvement: {Config.MIN_IMPROVEMENT}%")
    print(f"    Dramatic drop threshold: {Config.DRAMATIC_DROP}%")
    print(f"{'='*70}\n")

    # Early stopping state
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = None

    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch+1} (Best: {best_val_acc:.2f}% @ epoch {best_epoch+1}, patience: {epochs_without_improvement}/{Config.PATIENCE}) ---")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device, epoch, logger)

        # Evaluate
        val_loss, val_acc, per_letter_acc = evaluate(model, val_loader, criterion,
                                                     device, logger)

        # Get gradient and weight statistics
        grad_stats = logger.log_gradients(model, epoch)
        weight_stats = logger.log_weights(model, epoch)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch summary
        if (epoch + 1) % Config.DETAILED_LOG_INTERVAL == 0 or epoch == 0:
            logger.log_epoch_summary(epoch, train_loss, train_acc, val_loss, val_acc,
                                    grad_stats, weight_stats, per_letter_acc, current_lr)
        else:
            # Brief summary
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Step scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  [LR SCHEDULER] Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Log epoch to JSON
        grad_norms = {name: stats['norm'] for name, stats in grad_stats.items()}
        logger.log_epoch_json(epoch, train_loss, train_acc, val_loss, val_acc,
                             per_letter_acc, current_lr, grad_norms)

        # ========================================
        # EARLY STOPPING CHECKS
        # ========================================

        # Check for improvement
        if val_acc > best_val_acc + Config.MIN_IMPROVEMENT:
            improvement = val_acc - best_val_acc
            print(f"  [EARLY STOP] New best! {best_val_acc:.2f}% -> {val_acc:.2f}% (+{improvement:.2f}%)")
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  [EARLY STOP] No improvement for {epochs_without_improvement} epochs (best: {best_val_acc:.2f}%)")

        # Check for dramatic drop
        if val_acc < best_val_acc - Config.DRAMATIC_DROP:
            drop = best_val_acc - val_acc
            stop_reason = f"DRAMATIC DROP: Accuracy dropped {drop:.2f}% from best ({best_val_acc:.2f}% -> {val_acc:.2f}%)"
            print(f"\n{'!'*70}")
            print(f"  [EARLY STOP] {stop_reason}")
            print(f"{'!'*70}")
            break

        # Check for plateau (patience exhausted)
        if epochs_without_improvement >= Config.PATIENCE:
            stop_reason = f"PLATEAU: No improvement for {Config.PATIENCE} epochs (best: {best_val_acc:.2f}% @ epoch {best_epoch+1})"
            print(f"\n{'!'*70}")
            print(f"  [EARLY STOP] {stop_reason}")
            print(f"{'!'*70}")
            break

        # Check for target accuracy
        if val_acc >= 95:
            stop_reason = f"TARGET REACHED: {val_acc:.2f}% validation accuracy"
            print(f"\n{'*'*70}")
            print(f"  [EARLY STOP] {stop_reason}")
            print(f"{'*'*70}")
            break

        # Check for very low learning rate (stuck)
        if new_lr < 1e-7:
            stop_reason = f"LR EXHAUSTED: Learning rate too low ({new_lr:.2e})"
            print(f"\n{'!'*70}")
            print(f"  [EARLY STOP] {stop_reason}")
            print(f"{'!'*70}")
            break

    # Log final stopping reason
    if stop_reason is None:
        stop_reason = f"MAX EPOCHS: Completed all {Config.EPOCHS} epochs"

    print(f"\n{'='*70}")
    print(f"TRAINING STOPPED")
    print(f"{'='*70}")
    print(f"  Reason: {stop_reason}")
    print(f"  Total epochs: {epoch + 1}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
    print(f"{'='*70}")

    return model, stop_reason


# ================================================================
# MAIN
# ================================================================
def main():
    print("\n" + "="*70)
    print("ALPHABET GRADIENT DESCENT TRAINER")
    print("="*70)
    print("Task: Visual letter recognition (A-Z)")
    print("Method: Backpropagation with Cross-Entropy Loss")
    print("Challenge: Extremely constrained hidden layer (32 neurons)")
    print("="*70)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device(Config.DEVICE)
    print(f"\n[DEVICE] Using: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load fonts
    print(f"\n[CONFIG] Using --fonts={args.fonts}")
    font_loader = FontLoader(max_fonts=args.fonts)

    # Create renderer
    renderer = LetterRenderer(font_loader)

    # Create datasets
    print(f"\n{'='*60}")
    print("DATASET GENERATION")
    print(f"{'='*60}")
    train_dataset = AlphabetDataset(renderer, Config.VARIATIONS_PER_LETTER, is_training=True)
    val_dataset = AlphabetDataset(renderer, Config.VAL_VARIATIONS_PER_LETTER, is_training=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=0, pin_memory=True)

    print(f"[DATALOADER] Training batches: {len(train_loader)}")
    print(f"[DATALOADER] Validation batches: {len(val_loader)}")

    # Create model
    model = AlphabetNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=100)

    # Create logger
    logger = TrainingLogger()

    # Train
    model, stop_reason = train(model, train_loader, val_loader, criterion, optimizer,
                               scheduler, device, logger)

    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    # Reset confusion matrix for final eval
    logger.confusion_matrix = np.zeros((26, 26), dtype=int)
    final_loss, final_acc, final_per_letter = evaluate(model, val_loader, criterion,
                                                       device, logger)

    logger.print_final_summary()

    # Finalize JSON log
    logger.finalize_json(stop_reason, final_acc, final_loss, final_per_letter)

    # Save model and JSON log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = f"alphabet_gd_model_{timestamp}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_size': Config.HIDDEN_SIZE,
            'input_size': Config.INPUT_SIZE,
            'output_size': Config.OUTPUT_SIZE,
        },
        'final_accuracy': final_acc,
        'final_loss': final_loss,
    }, model_path)
    print(f"[SAVE] Model saved to: {model_path}")

    # Save JSON training log
    json_path = f"alphabet_gd_log_{timestamp}.json"
    logger.save_json(json_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Final Validation Accuracy: {final_acc:.2f}%")
    print(f"  Random Baseline: {100/26:.2f}%")
    print(f"  Improvement: {final_acc - 100/26:.2f}%")
    print(f"")
    print(f"  Saved files:")
    print(f"    Model: {model_path}")
    print(f"    Log:   {json_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
