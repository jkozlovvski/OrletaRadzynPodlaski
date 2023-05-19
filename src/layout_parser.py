from torch.utils.data import Dataset

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import layoutparser as lp
import cv2


class LayoutParserTransform:
    def __init__(self, model):
        self.model = model
        self.pooling_layer = nn.AvgPool2d(kernel_size=8, stride=8)

    def __call__(self, sample):
        image = np.array(sample)

        imge_out = Image.fromarray(image.astype('uint8'))
        image = imge_out.convert("RGB")
        image = np.array(image)


        # Użyj modelu do wykrycia layoutu
        layout = self.model.detect(image)

        # Stwórz puste tło
        blank_image = np.zeros_like(image)

        # Narysuj bounding boxy
        image_with_boxes_only = lp.draw_box(blank_image, layout, box_width=3)
        image_with_boxes_only_np = np.array(image_with_boxes_only).astype('uint8')


        # # Przekonwertuj na skalę szarości
        
        # image_with_boxes_only_gray = cv2.cvtColor(image_with_boxes_only_np, cv2.COLOR_BGR2GRAY)

        # # Przekształć obraz ndarray do tensora
        # image_with_boxes_only_gray = torch.from_numpy(image_with_boxes_only_gray)

        # # Dodaj dodatkowy wymiar dla kanałów
        # image_with_boxes_only_gray = image_with_boxes_only_gray.unsqueeze(0)

        # Wykonanie operacji pooling
        image_with_boxes_only_tensor = torch.tensor(image_with_boxes_only_np, dtype=torch.float32)
        image_pooled = self.pooling_layer(image_with_boxes_only_tensor)

        # Usuń niepotrzebne wymiary
        image_pooled = torch.squeeze(image_pooled)

        return image_pooled


class ResizeWithPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if not isinstance(sample, np.ndarray):
            image = np.array(sample)
        else:
            image = sample

        # Obliczamy różnicę między rozmiarem docelowym a rzeczywistym rozmiarem obrazka
        diff = self.size - min(image.shape[0], image.shape[1])

        # Dokonujemy paddingu tylko wtedy, gdy jest to konieczne
        if diff > 0:
            padding = diff // 2
            image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

        # Skalujemy obraz do rozmiaru docelowego
        image = cv2.resize(image, (self.size, self.size))

        return image


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # # Przekonwertuj obrazek na PIL Image jeżeli jest w formacie NumPy array
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label