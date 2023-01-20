from itertools import product
from queue import Queue
from typing import Union

import numpy as np
import torch
from skimage import segmentation

from autoexplainer.utils import torch_image_to_numpy_image


def batch_segmentation(
    x_batch: torch.Tensor, mask_function_name: str, **kwargs: Union[int, str, float]
) -> torch.Tensor:
    assert mask_function_name in MASK_NAME_TO_FUNCTION.keys()
    masks_list = []
    mask_func = MASK_NAME_TO_FUNCTION[mask_function_name]
    for image in x_batch:
        image_np = torch_image_to_numpy_image(image)
        m = mask_func(image_np, **kwargs)
        masks_list.append(m)

    masks = torch.tensor(np.array(masks_list)).unsqueeze(1).to(x_batch.device)

    return masks


def kandinsky_figures_segmentation(  # type: ignore
    img,
    bg=(0.5882353187, 0.5882353187, 0.5882353187),
    tolerance=(4, 4, 4),
    bg_tolerance=(0.5, 0.5, 0.5),
    **kwargs: Union[int, str, float],
):
    mask = -np.ones((img.shape[0], img.shape[1]))
    mask[
        (abs(img[:, :, 0] - bg[0]) < bg_tolerance[0])
        & (abs(img[:, :, 1] - bg[1]) < bg_tolerance[1])
        & (abs(img[:, :, 2] - bg[2]) < bg_tolerance[2])
    ] = 0

    def get_neighbours(x, y):  # type: ignore
        neighbours_coords = []
        if x - 1 > 0:
            neighbours_coords.append((x - 1, y))
        if y - 1 > 0:
            neighbours_coords.append((x, y - 1))
        if x + 1 < img.shape[0]:
            neighbours_coords.append((x + 1, y))
        if y + 1 < img.shape[1]:
            neighbours_coords.append((x, y + 1))
        return neighbours_coords

    def get_color(x, y):  # type: ignore
        return float(img[x][y][0]), float(img[x][y][1]), float(img[x][y][2])

    def is_close(col1, col2):  # type: ignore
        return all(np.absolute(np.array(col1) - np.array(col2)) < np.array(tolerance))

    def fill(x, y, color):  # type: ignore
        start_color = get_color(x, y)
        q: Queue = Queue()
        q.put((x, y))
        while not q.empty():
            x, y = q.get()
            if mask[x][y] < 0:
                mask[x][y] = color
            neighbours = get_neighbours(x, y)
            for nx, ny in neighbours:
                if is_close(get_color(nx, ny), start_color) and mask[nx][ny] == -1:
                    q.put((nx, ny))
                    mask[nx][ny] = -2

    i = 1
    for x, y in product(range(img.shape[0]), range(img.shape[1])):
        if mask[x, y] < 0:
            fill(x, y, i)
            i += 1
    return mask.astype(np.longlong)


MASK_NAME_TO_FUNCTION = {
    "slic": segmentation.slic,
    "kandinsky_figures": kandinsky_figures_segmentation,
}
