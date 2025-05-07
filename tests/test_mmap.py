import torch
from src.data.tiered_loader import TieredLoader

def test_mmap_loading():
    loader = TieredLoader(
        ssd_path="src/data/processed/mmap_datasets",
        hdd_path="src/data/processed/mmap_datasets"  # Using same dir for test
    )
     
    for i in range(10):
        batch = loader.fetch(i)
        assert batch.shape == (256, 256), f"Bad shape for batch {i}"
        assert isinstance(batch, torch.Tensor), "Not a tensor"
     
    print("✅ All MMAP loads validated")

if __name__ == "__main__":
    test_mmap_loading()
