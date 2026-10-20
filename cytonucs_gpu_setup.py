"""
GPU Configuration for CytoNucs StarDist Training

This module sets up GPU memory management to prevent OOM errors
and ensures stable training on GPU.
"""
import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def setup_gpu(memory_limit=None, allow_growth=True, visible_devices=None):
    """
    Configure GPU for training.

    Args:
        memory_limit: int, maximum GPU memory in MB (None = unlimited)
        allow_growth: bool, allow dynamic memory allocation
        visible_devices: str, comma-separated GPU indices (e.g., "0,1")

    Returns:
        list of available GPU devices
    """
    # Set visible devices if specified
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_devices)

    # The paper-run queue caps a job that shares its GPU (GPU_MEMORY_LIMIT_MB); a limit in the config wins.
    if memory_limit is None and int(os.environ.get('GPU_MEMORY_LIMIT_MB', 0) or 0):
        memory_limit = int(os.environ['GPU_MEMORY_LIMIT_MB'])

    # Get list of GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("\n⚠️  No GPU detected - training will run on CPU")
        print("   This will be significantly slower than GPU training.")
        return []

    print(f"\n✅ Detected {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")

    try:
        # Configure each GPU
        for gpu in gpus:
            if allow_growth and memory_limit is None:
                # Allow dynamic memory allocation
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✓ Enabled memory growth for {gpu.name}")

            if memory_limit is not None:
                # Set memory limit
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                print(f"   ✓ Set memory limit to {memory_limit} MB for {gpu.name}")

        # Verify logical devices
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"\n✅ Configured {len(logical_gpus)} logical GPU(s)")

        return gpus

    except RuntimeError as e:
        print(f"\n❌ Error configuring GPU: {e}")
        print("   Virtual devices must be set before GPUs have been initialized")
        return []


def print_gpu_info():
    """Print detailed GPU information"""
    print("\n" + "=" * 60)
    print("GPU Information")
    print("=" * 60)

    # Physical devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

    # Logical devices
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"\nLogical GPUs: {len(logical_gpus)}")
    for i, gpu in enumerate(logical_gpus):
        print(f"  Logical GPU {i}: {gpu}")

    # CUDA info
    print(f"\nCUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU support: {len(tf.config.list_physical_devices('GPU')) > 0}")

    print("=" * 60 + "\n")


def auto_select_device(prefer_gpu=True):
    """
    Automatically select best available device.

    Args:
        prefer_gpu: bool, prefer GPU over CPU if available

    Returns:
        device string (e.g., "/GPU:0" or "/CPU:0")
    """
    gpus = tf.config.list_physical_devices('GPU')

    if gpus and prefer_gpu:
        return f"/GPU:0"
    else:
        return "/CPU:0"


if __name__ == "__main__":
    # Test GPU setup
    print("Testing GPU configuration...")
    setup_gpu(allow_growth=True)
    print_gpu_info()
    device = auto_select_device()
    print(f"Selected device: {device}")
