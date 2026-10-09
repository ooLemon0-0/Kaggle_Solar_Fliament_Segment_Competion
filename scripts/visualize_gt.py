import _bootstrap
import argparse
import random
from PIL import Image
from src.data.dataset import FilamentDataset, discover, mask_from_annotation
from src.utils.common import ROOT
from src.utils.visualization import panel, join_panels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=30)
    args = parser.parse_args()
    dataset = FilamentDataset(discover()[0])
    images = random.Random(42).sample(dataset.images, min(args.count, len(dataset)))
    for im in images:
        with Image.open(dataset.image_dir / im['file_name']) as source:
            gray = source.convert('L')
        anns = dataset.annotations[im['id']]
        masks = [mask_from_annotation(a, im) for a in anns]
        join_panels([panel(gray, [], im['file_name']), panel(gray, masks, f"{im['id']} | GT {len(masks)}", anns)], ROOT / f"output/data_analysis/gt/{im['id']}.png")
    print(f'Generated {len(images)} GT visualizations')
