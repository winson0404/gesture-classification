from typing import List, Tuple, Optional, Dict
from torch import Tensor, nn


class Compose:
    def __init__(self, transforms: List[nn.Module]):
        self.transforms = transforms

    def __call__(self, image, target) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target