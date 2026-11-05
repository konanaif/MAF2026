import os
import json
import torch
from pathlib import Path
from torchvision import datasets, transforms

MAF_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = str(MAF_ROOT / 'data' / 'casual_path_tracing')
KNOWNS_URL = "https://rome.baulab.info/data/dsets/known_1000.json"
IMAGENET_ROOT = SAMPLE_DIR + '/imagenet'


class KnownsDataset(torch.utils.data.Dataset):
    """ROME known_1000 factual statements: [{"prompt", "subject", "attribute", ...}, ...]"""
    def __init__(self, data_dir: str = SAMPLE_DIR, reverse=False, *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.known_loc = self.data_dir / "known_1000.json"
        if not self.known_loc.exists():
            print('The data file [[ {} ]] is not in directory'.format(self.known_loc))
            print('It will be downloaded...')
            self.download()

        with open(self.known_loc, "r") as f:
            self.data = json.load(f)
        if reverse:
            self.data = self.data[::-1]
        print(f"Loaded dataset with {len(self)} elements")

    def download(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        torch.hub.download_url_to_file(KNOWNS_URL, self.known_loc)
        print('Downloaded successfully', end='\n\n')

    def to_dataset(self):
        result = {
            'line_data': self,
            'prompt': [d["prompt"] for d in self.data],
            'subject': [d["subject"] for d in self.data],
            'attribute': [d["attribute"] for d in self.data],
        }
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ImageNetDataset(datasets.ImageNet):
    """ImageNet-1k (torchvision layout: <root>/meta.bin, <root>/<split>/<wnid>/*.JPEG) with standard ViT evaluation transforms"""
    num_classes = 1000

    def __init__(self, root: str = IMAGENET_ROOT, split: str = 'val'):
        # ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        # Standard ImageNet validation transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        if not os.path.isfile(os.path.join(root, 'meta.bin')) or not os.path.isdir(os.path.join(root, split)):
            self.download(root)
        super(ImageNetDataset, self).__init__(root=root, split=split, transform=transform)

    def download(self, root):
        # ImageNet cannot be downloaded automatically (license); prepare it in the torchvision layout
        raise FileNotFoundError(
            'ImageNet is not found in [[ {} ]]. Prepare <root>/meta.bin and <root>/val/<wnid>/*.JPEG '
            '(prepare it with: python -m MAF.datamodule.casual_path_tracing.prepare_imagenet '
            '--src <parquet dir> --dst <root>)'.format(root))

    def to_dataset(self):
        result = {
            'dataset': self,
            'num_classes': self.num_classes,
            'target': self.targets,
        }
        return result
