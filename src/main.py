"""main
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.config import CFG
from src.data_preparation import create_labels_map
from src.model import SorghumModel


def process_image(img_path, transforms=CFG.data_transforms, display=False):
    """Process Image

    Args:
        img_path (str): image path
        transforms (_type_, optional): data transform. Defaults to data_transforms.
        display (bool, optional): Display image or not. Defaults to False.

    Returns:
        _type_: return torch.tensor image
    """
    img = Image.open(img_path)
    if display:
        img.show()
    img = np.array(img)
    aug = transforms(image=img)
    img = aug['image']
    img = img[None, :]
    img = img.to(CFG.device)
    return img


def predict(model, img, labels_map):
    """Predict result from image

    Args:
        model (_type_): input model
        img (_type_): processed image
        labels_map (_type_): decode result to label

    Returns:
        _type_: prediction
    """
    softmax = nn.Softmax(dim=1)
    embeddings = model.extract(img)
    output = softmax(CFG.S * F.linear(F.normalize(embeddings),
                                      F.normalize(model.fc.weight))).cpu().detach().numpy()
    probs = output[0]
    output = np.argmax(output, -1)
    prediction = labels_map[output[0]]
    return prediction, probs


if __name__ == "__main__":
    model = SorghumModel(CFG.backbone, CFG.embedding_size, CFG.num_classes)
    model.load_state_dict(torch.load(CFG.model_path, map_location="cpu"))
    model.eval()

    IMAGE_PATH = "data/2017-06-01__10-26-27-479.jpeg"
    image = process_image(img_path=IMAGE_PATH,
                          transforms=CFG.data_transforms, display=False)
    labels_map = create_labels_map()

    # print(predict(model, image, labels_map))
    preds, probs = predict(model, image, labels_map)
    print(sum(probs))
