from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import os
import torch

class IDCardDataProcessor(Dataset):
    def __init__(self, image_dir, annotations_file, image_size, transform=None):
        self.image_dir = image_dir
        self.annotations = json.load(open(annotations_file))
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = self.annotations[index]['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path))
        # image = image.convert('RGB')
        original_size = image.size
        # print(original_size)

        # presense:
        presense = self.annotations[index]['has_symbol']
        presense_tensor = torch.tensor([presense], dtype=torch.float32)

        # bbox:
        if presense:
            bbox = self.annotations[index]['bbox']

            # Dont do this, Unless you want to change dataset, currently I'm using the resized images, 
            # so you dont need to manually change the bbox coordinates 
            # adjusted_bbox = adjust_bounding_boxes([bbox], original_size, (self.image_size, self.image_size))[0]
            # x_min, y_min, x_max, y_max = adjusted_bbox


            x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
            bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            bbox_tensor = torch.zeros(4, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        
        # print(image.size())
        return image, presense_tensor, bbox_tensor

def adjust_bounding_boxes(bboxes, original_size, target_size):
    original_width, original_height = original_size
    target_width, target_height = target_size
    
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min = bbox['x_min']
        y_min = bbox['y_min']
        x_max = bbox['x_max']
        y_max = bbox['y_max']
        x_min_adj = x_min * scale_width
        y_min_adj = y_min * scale_height
        x_max_adj = x_max * scale_width
        y_max_adj = y_max * scale_height
        
        adjusted_bboxes.append((x_min_adj, y_min_adj, x_max_adj, y_max_adj))
    
    return adjusted_bboxes  
