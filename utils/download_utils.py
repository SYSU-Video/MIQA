import requests
from tqdm import tqdm
from typing import Optional, List
from pathlib import Path
import torch
import gdown

def download_checkpoint(
        url: str,
        save_path: str,
        chunk_size: int = 8192,
        verify_ssl: bool = True,
        max_retries: int = 3
) -> bool:
    """
    Download checkpoint file from URL with progress bar and retry mechanism.

    Args:
        url (str): URL to download the checkpoint from
        save_path (str): Local path to save the checkpoint
        chunk_size (int): Size of chunks to download at a time (bytes)
        verify_ssl (bool): Whether to verify SSL certificates
        max_retries (int): Maximum number of download retries

    Returns:
        bool: True if download successful, False otherwise
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(f"Downloading checkpoint from: {url}")
            print(f"Saving to: {save_path}")

            # Send GET request with stream=True for large files
            response = requests.get(url, stream=True, verify=verify_ssl, timeout=30)
            response.raise_for_status()

            # Get total file size from headers
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(save_path, 'wb') as f, tqdm(
                    desc=f"Downloading (Attempt {attempt + 1}/{max_retries})",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify the downloaded file is not empty
            if save_path.stat().st_size == 0:
                raise ValueError("Downloaded file is empty")

            print(f"✓ Download completed successfully!")
            return True

        except Exception as e:
            print(f"✗ Download attempt {attempt + 1} failed: {str(e)}")
            if save_path.exists():
                save_path.unlink()  # Remove partial download

            if attempt == max_retries - 1:
                print(f"✗ Failed to download after {max_retries} attempts")
                return False

            print(f"Retrying...")

    return False


def verify_checkpoint(checkpoint_path: str) -> bool:
    """
    Verify checkpoint file integrity.

    Args:
        checkpoint_path (str): Path to checkpoint file

    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return False

    # Check file size
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint size: {file_size_mb:.2f} MB")

    # Try to load checkpoint to verify it's a valid PyTorch file
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint is a valid PyTorch file")

        # Print checkpoint info
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                print(f"Number of parameters: {len(checkpoint['state_dict'])}")

        return True

    except Exception as e:
        print(f"✗ Invalid checkpoint file: {str(e)}")
        return False


def ensure_checkpoint(
        checkpoint_path: str,
        download_urls: List[str],
        force_download: bool = False
) -> bool:
    """
    Ensure checkpoint exists, download if necessary.

    Args:
        checkpoint_path (str): Path where checkpoint should be located
        download_urls (List[str]): List of URLs to try downloading from (in order)
        force_download (bool): Force re-download even if file exists

    Returns:
        bool: True if checkpoint is available, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)

    # Check if checkpoint already exists and is valid
    if checkpoint_path.exists() and not force_download:
        print(f"Checkpoint found at: {checkpoint_path}")
        if verify_checkpoint(checkpoint_path):
            return True
        else:
            print("Existing checkpoint is invalid, will re-download...")
            checkpoint_path.unlink()

    # Try downloading from each URL
    for i, url in enumerate(download_urls):
        print(f"\nAttempting download from source {i + 1}/{len(download_urls)}...")
        if download_checkpoint(url, checkpoint_path):
            if verify_checkpoint(checkpoint_path):
                return True
            else:
                print("Downloaded file verification failed, trying next source...")
                checkpoint_path.unlink()

    print(f"\n✗ Failed to download checkpoint from all sources")
    return False


def ensure_checkpoint_with_gdown(
        checkpoint_path: str,
        gdrive_file_ids: List[str],
        force_download: bool = False
) -> bool:
    checkpoint_path = Path(checkpoint_path)

    if force_download and checkpoint_path.exists():
        print(f"Force download enabled. Removing existing file: {checkpoint_path}")
        checkpoint_path.unlink()

    if checkpoint_path.exists():
        print(f"Checkpoint found at: {checkpoint_path}")
        if verify_checkpoint(checkpoint_path):
            return True
        else:
            print("Existing checkpoint is invalid, will re-download...")
            checkpoint_path.unlink()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(gdrive_file_ids):
        print(f"\nAttempting download from Google Drive (source {i + 1}/{len(gdrive_file_ids)})...")
        print(f"File ID: {file_id}")

        try:
            gdown.download(id=file_id, output=str(checkpoint_path), quiet=False)

            if verify_checkpoint(checkpoint_path):
                return True
            else:
                print("Downloaded file verification failed. Trying next source...")
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

        except Exception as e:
            print(f"✗ Download failed for ID '{file_id}'. Error: {e}")
            print("Trying next source...")

            if checkpoint_path.exists():
                checkpoint_path.unlink()

    print(f"\n✗ Failed to download and verify checkpoint from all Google Drive sources.")
    return False