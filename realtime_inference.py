import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Deque
from collections import OrderedDict, deque
import time
from datetime import datetime
import numpy as np
import cv2
import threading
from queue import Queue

# Image processing imports
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')  # Use interactive backend for real-time plotting

# Import your existing model components
from models.MIQA_base import get_torch_model, get_timm_model
from models.RA_MIQA import RegionVisionTransformer
from utils.download_utils import ensure_checkpoint

# Model configuration
MODEL_CONFIGS = {
    'cls': {
        'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
        'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
        'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
        'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
    },
    'det': {
        'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
        'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
        'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
        'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
    },
    'ins': {
        'resnet50': [f"https://drive.google.com/uc?id={file_id}"],
        'efficientnet_b5': [f"https://drive.google.com/uc?id={file_id}"],
        'vit_small_patch16_224': [f"https://drive.google.com/uc?id={file_id}"],
        'ra_miqa': [f"https://drive.google.com/uc?id={file_id}"]
    }
}


class RealTimeQualityMonitor:
    """
    Real-time webcam quality assessment system with live visualization.

    This system captures frames from your webcam, evaluates their quality using
    the MIQA model, and displays both the live video feed with quality scores
    overlaid and a dynamic chart showing quality trends over time.
    """

    def __init__(self, task: str, model_name: str = 'ra_miqa',
                 metric_type: str = 'composite', device: Optional[str] = None,
                 camera_id: int = 0, fps_target: int = 10,
                 history_window: int = 100):
        """
        Initialize the real-time quality monitoring system.

        Args:
            task: Task type - 'cls', 'det', or 'ins'
            model_name: Model architecture to use
            metric_type: Training objective
            device: Device to run inference on
            camera_id: Camera device ID (usually 0 for default webcam)
            fps_target: Target frame rate for quality assessment
            history_window: Number of recent scores to keep in the visualization
        """
        self.task = task.lower()
        self.model_name = model_name
        self.metric_type = metric_type
        self.camera_id = camera_id
        self.fps_target = fps_target
        self.history_window = history_window

        # Setup logging
        self.logger = self._setup_logger()

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info("🎥 Initializing Real-Time Quality Monitor")
        self.logger.info(f"   Task: {self.task.upper()}")
        self.logger.info(f"   Model: {self.model_name}")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Target FPS: {self.fps_target}")

        # Validate and load model
        self._validate_config()
        self.model = self._load_model()
        self.transforms1, self.transforms2 = self._get_transforms()

        # Initialize data structures for real-time processing
        # These deques store recent timestamps and quality scores for the chart
        self.timestamps: Deque[float] = deque(maxlen=history_window)
        self.quality_scores: Deque[float] = deque(maxlen=history_window)

        # Thread-safe queue for passing frames between capture and processing threads
        self.frame_queue = Queue(maxsize=2)

        # Shared state for current frame and score (with thread lock for safety)
        self.current_frame = None
        self.current_score = None
        self.frame_lock = threading.Lock()

        # Control flags for graceful shutdown
        self.running = False
        self.capture_thread = None
        self.processing_thread = None

        # Performance tracking
        self.start_time = None
        self.frame_count = 0
        self.actual_fps = 0

        self.logger.info("✅ System ready for real-time monitoring\n")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with minimal output for real-time use."""
        logger = logging.getLogger('MIQA_RealTime')
        logger.setLevel(logging.INFO)
        logger.handlers = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.task not in MODEL_CONFIGS:
            raise ValueError(f"Invalid task '{self.task}'")
        if self.model_name not in MODEL_CONFIGS[self.task]:
            raise ValueError(f"Model '{self.model_name}' not available")

    def _get_checkpoint_path(self) -> str:
        """Generate checkpoint path."""
        base_dir = Path('models') / 'checkpoints' / f'miqa_{self.task}'
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.model_name}_mse_miqa_{self.task}_{self.metric_type}_best.pth.tar"
        return str(base_dir / filename)

    def _download_weights(self, checkpoint_path: str) -> bool:
        """Download model weights if needed."""
        if os.path.exists(checkpoint_path):
            self.logger.info("✓ Found cached model weights")
            return True

        self.logger.info("⏬ Downloading model weights...")
        download_urls = MODEL_CONFIGS[self.task][self.model_name]

        if download_urls[0] == f"https://drive.google.com/uc?id={file_id}":
            self.logger.error("❌ Download URLs not configured")
            return False

        return ensure_checkpoint(checkpoint_path, download_urls)

    def _create_model(self) -> torch.nn.Module:
        """Create model architecture."""
        if self.model_name == 'ra_miqa':
            model = RegionVisionTransformer(
                base_model_name='vit_small_patch16_224',
                pretrained=False,
                mmseg_config_path='models/model_configs/fcn_sere-small_finetuned_fp16_8x32_224x224_3600_imagenets919.py',
                checkpoint_path='models/checkpoints/sere_finetuned_vit_small_ep100.pth'
            )
        else:
            try:
                model = get_torch_model(model_name=self.model_name, pretrained=False, num_classes=1)
            except Exception:
                model = get_timm_model(model_name=self.model_name, pretrained=False, num_classes=1)
        return model

    def _load_model(self) -> torch.nn.Module:
        """Load model with weights."""
        checkpoint_path = self._get_checkpoint_path()

        if not self._download_weights(checkpoint_path):
            raise RuntimeError("Cannot proceed without model weights")

        self.logger.info("🔧 Loading model...")
        model = self._create_model()

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        self.logger.info("✓ Model loaded successfully")
        return model

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get image preprocessing transforms."""
        transforms_list1 = [
            transforms.Resize(288),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        transform_list_2 = [
            transforms.Resize(288),
            transforms.CenterCrop((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transforms_list1), transforms.Compose(transform_list_2)

    def _prepare_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a video frame for model input.

        This converts the raw BGR frame from OpenCV into the normalized RGB tensors
        that the model expects, applying the same preprocessing used during training.
        """
        # Convert BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for transforms
        img = Image.fromarray(frame_rgb)

        # Apply preprocessing transforms
        img1 = self.transforms1(img).unsqueeze(0)
        img2 = self.transforms2(img).unsqueeze(0)

        return img1, img2

    def _capture_frames(self):
        """
        Capture thread: Continuously grabs frames from webcam.

        This runs in a separate thread to ensure frame capture isn't blocked by
        the model inference, which can be slower. Frames are placed in a queue
        for the processing thread to consume.
        """
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            self.logger.error(f"❌ Cannot open camera {self.camera_id}")
            self.running = False
            return

        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.logger.info(f"✓ Camera {self.camera_id} opened successfully")

        # Calculate target delay between frames based on desired FPS
        frame_delay = 1.0 / self.fps_target
        last_capture_time = time.time()

        while self.running:
            current_time = time.time()

            # Throttle capture to target FPS
            if current_time - last_capture_time < frame_delay:
                time.sleep(0.001)  # Short sleep to avoid busy waiting
                continue

            ret, frame = cap.read()

            if not ret:
                self.logger.warning("⚠️  Failed to capture frame")
                continue

            # Try to add frame to queue (non-blocking)
            # If queue is full, we skip this frame to avoid blocking
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            last_capture_time = current_time

        cap.release()
        self.logger.info("✓ Camera released")

    @torch.no_grad()
    def _process_frames(self):
        """
        Processing thread: Runs quality assessment on captured frames.

        This thread takes frames from the queue, runs them through the MIQA model,
        and updates the shared state with results. Running inference in a separate
        thread prevents it from blocking the capture or visualization.
        """
        while self.running:
            # Get frame from queue (blocks if queue is empty)
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()

            try:
                # Preprocess frame
                img_cropped, img_resized = self._prepare_frame(frame)

                # Move to device
                img_cropped = img_cropped.to(self.device)
                img_resized = img_resized.to(self.device)

                # Run inference
                output = self.model(img_cropped, img_resized)
                score = output.item()

                # Calculate elapsed time for timestamp
                elapsed_time = time.time() - self.start_time

                # Update shared state with thread lock
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_score = score
                    self.timestamps.append(elapsed_time)
                    self.quality_scores.append(score)
                    self.frame_count += 1

                # Update FPS calculation
                if elapsed_time > 0:
                    self.actual_fps = self.frame_count / elapsed_time

            except Exception as e:
                self.logger.warning(f"⚠️  Error processing frame: {e}")

    def _draw_overlay(self, frame: np.ndarray, score: float) -> np.ndarray:
        """
        Add quality score overlay to the frame.

        This creates a colored box in the corner showing the current quality score.
        The color changes based on score: red for low quality, green for high quality.
        """
        # Create a copy to avoid modifying the original
        display_frame = frame.copy()

        # Normalize score to [0, 1] for color mapping (adjust range as needed)
        # Assuming scores roughly range from 0 to 100
        norm_score = score / 100.0
        norm_score = max(0, min(1, norm_score))

        # Color interpolation: red -> yellow -> green
        if norm_score < 0.5:
            # Red to yellow
            r, g, b = 255, int(255 * norm_score * 2), 0
        else:
            # Yellow to green
            r, g, b = int(255 * (2 - norm_score * 2)), 255, 0

        color_bgr = (b, g, r)  # OpenCV uses BGR

        # Draw semi-transparent background box
        overlay = display_frame.copy()
        box_coords = (10, 10, 280, 70)
        cv2.rectangle(overlay, box_coords[:2], box_coords[2:], color_bgr, -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Add text with quality score
        score_text = f"Quality: {score:.2f}"
        cv2.putText(display_frame, score_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Add FPS counter
        fps_text = f"FPS: {self.actual_fps:.1f}"
        cv2.putText(display_frame, fps_text, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return display_frame

    def _update_plot(self, frame_num):
        """
        Animation callback for updating the quality trend chart.

        This is called by matplotlib's animation system to update the plot with
        the latest quality scores. It runs on the main thread along with the
        OpenCV window display.
        """
        # Get current data with thread lock
        with self.frame_lock:
            if len(self.timestamps) == 0:
                return

            timestamps = list(self.timestamps)
            scores = list(self.quality_scores)
            current_score = self.current_score

        # Clear and redraw the plot
        self.ax.clear()

        # Plot quality score over time
        if len(timestamps) > 1:
            self.ax.plot(timestamps, scores, linewidth=2, color='#2E86AB',
                         marker='o', markersize=3, markerfacecolor='white',
                         markeredgewidth=1)

            # Add current score indicator
            if current_score is not None:
                self.ax.axhline(y=current_score, color='#A23B72',
                                linestyle='--', linewidth=1.5, alpha=0.6)

        # Styling
        self.ax.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Quality Score', fontsize=10, fontweight='bold')
        self.ax.set_title('Real-Time Quality Assessment', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics text
        if len(scores) > 0:
            avg_score = np.mean(scores)
            stats_text = f"Current: {current_score:.2f}\nAverage: {avg_score:.2f}\nFrames: {len(scores)}"
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Auto-scale y-axis with some padding
        if len(scores) > 0:
            y_min, y_max = min(scores), max(scores)
            y_range = y_max - y_min
            padding = max(y_range * 0.1, 1.0)
            self.ax.set_ylim(y_min - padding, y_max + padding)

    def start_monitoring(self, save_log: bool = False, log_file: str = 'quality_log.txt'):
        """
        Start the real-time quality monitoring system.

        This launches all three components:
        1. Capture thread (grabs frames from webcam)
        2. Processing thread (runs MIQA inference)
        3. Main thread (displays results and manages visualization)

        Args:
            save_log: Whether to save quality scores to a log file
            log_file: Path to log file if saving
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING REAL-TIME QUALITY MONITORING")
        self.logger.info("=" * 60)
        self.logger.info("Press 'q' in the video window to quit")
        self.logger.info("=" * 60 + "\n")

        # Initialize timing
        self.start_time = time.time()
        self.running = True

        # Start capture and processing threads
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)

        self.capture_thread.start()
        self.processing_thread.start()

        # Setup matplotlib figure for quality chart
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.15)

        # Create animation for updating the chart
        # This updates the plot every 100ms
        ani = FuncAnimation(self.fig, self._update_plot, interval=100, cache_frame_data=False)

        # Setup log file if requested
        log_handle = None
        if save_log:
            log_handle = open(log_file, 'w')
            log_handle.write("timestamp,quality_score\n")
            self.logger.info(f"📝 Logging quality scores to {log_file}")

        try:
            # Main display loop
            plt.ion()  # Turn on interactive mode
            plt.show()

            while self.running:
                # Get current frame and score
                with self.frame_lock:
                    if self.current_frame is not None and self.current_score is not None:
                        display_frame = self._draw_overlay(self.current_frame, self.current_score)

                        # Log if enabled
                        if log_handle and len(self.timestamps) > 0:
                            log_handle.write(f"{self.timestamps[-1]:.3f},{self.current_score:.4f}\n")
                            log_handle.flush()
                    else:
                        # Create placeholder frame while waiting for first result
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "Initializing...", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display the frame in OpenCV window
                cv2.imshow('Real-Time Quality Assessment', display_frame)

                # Update matplotlib plot
                plt.pause(0.001)

                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("\n🛑 Quit command received")
                    break

        except KeyboardInterrupt:
            self.logger.info("\n🛑 Interrupted by user")

        finally:
            # Cleanup
            self.running = False

            # Wait for threads to finish
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            if self.processing_thread:
                self.processing_thread.join(timeout=2)

            # Close windows and files
            cv2.destroyAllWindows()
            plt.close('all')

            if log_handle:
                log_handle.close()
                self.logger.info(f"✓ Log saved to {log_file}")

            # Print final statistics
            self._print_summary()

    def _print_summary(self):
        """Print final statistics after monitoring session."""
        with self.frame_lock:
            scores = list(self.quality_scores)

        if len(scores) == 0:
            self.logger.info("No frames processed")
            return

        total_time = time.time() - self.start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {total_time:.1f} seconds")
        self.logger.info(f"Frames processed: {len(scores)}")
        self.logger.info(f"Average FPS: {len(scores) / total_time:.2f}")
        self.logger.info(f"\nQuality Statistics:")
        self.logger.info(f"  Average: {np.mean(scores):.2f}")
        self.logger.info(f"  Min: {np.min(scores):.2f}")
        self.logger.info(f"  Max: {np.max(scores):.2f}")
        self.logger.info(f"  Std Dev: {np.std(scores):.2f}")
        self.logger.info("=" * 60 + "\n")


def main():
    """Command-line interface for real-time quality monitoring."""
    parser = argparse.ArgumentParser(
        description='MIQA Real-Time Webcam Quality Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        This tool opens your webcam and continuously evaluates image quality in real-time.
        It displays two windows:
          1. Video feed with quality score overlay
          2. Dynamic chart showing quality trends over time
        
        Press 'q' in the video window to stop monitoring.
        
        Example:
          python realtime_inference.py --task cls --fps 5 --save-log
                """
    )

    parser.add_argument('--task', type=str, required=True,
                        choices=['cls', 'det', 'ins'],
                        help='Task type')
    parser.add_argument('--model', type=str, default='ra_miqa',
                        choices=['resnet50', 'efficientnet_b5', 'vit_small_patch16_224', 'ra_miqa'],
                        help='Model architecture')
    parser.add_argument('--metric-type', type=str, default='composite',
                        choices=['composite', 'consistency', 'accuracy'],
                        help='Training metric type')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run on (auto-detect if not specified)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Target frames per second for quality assessment (default: 10)')
    parser.add_argument('--history', type=int, default=100,
                        help='Number of recent scores to show in chart (default: 100)')
    parser.add_argument('--save-log', action='store_true',
                        help='Save quality scores to log file')
    parser.add_argument('--log-file', type=str, default='quality_log.txt',
                        help='Log file path (default: quality_log.txt)')

    args = parser.parse_args()

    try:
        # Initialize the monitoring system
        monitor = RealTimeQualityMonitor(
            task=args.task,
            model_name=args.model,
            metric_type=args.metric_type,
            device=args.device,
            camera_id=args.camera,
            fps_target=args.fps,
            history_window=args.history
        )

        # Start monitoring
        monitor.start_monitoring(save_log=args.save_log, log_file=args.log_file)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
    '''
    python realtime_inference.py --task cls --fps 5

    python realtime_inference.py --task det --fps 10 --save-log --log-file my_session.txt
    '''