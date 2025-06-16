from .kitti_dataset import KITTIRAWDataset, KITTIOdomDataset, KITTIDepthDataset
from .mono_dataset import MonoDataset

# Create a simple CityscapesDataset class for compatibility
class CityscapesDataset(MonoDataset):
    """Cityscapes dataset - placeholder for compatibility"""
    def __init__(self, *args, **kwargs):
        super(CityscapesDataset, self).__init__(*args, **kwargs)
        self.K = None  # Will be set in the dataset
