import numpy as np
from PIL import Image
def rle2mask(mask_rle: str, shape=None, label: int = 0):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    rle = np.array(list(map(int, mask_rle.split())))
    labels = np.zeros(shape).flatten()

    for start, end in zip(rle[::2], rle[1::2]):
        labels[start:start + end] = label

    return labels.reshape(shape).T  # Needed to align to RLE direction


def mask_to_rle(mask):
    # Rescale image to original size
    size = int(len(mask.flatten()) ** .5)
    n = Image.fromarray(mask.reshape((size, size)) * 255.0)
    n = np.array(n).astype(np.float32)
    # Get pixels to flatten
    pixels = n.T.flatten()
    # Round the pixels using the half of the range of pixel value
    pixels = (pixels - min(pixels) > ((max(pixels) - min(pixels)) / 2)).astype(int)
    pixels = np.nan_to_num(pixels)  # incase of zero-div-error

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def create_masks(df):
    for i in df.index:
        mask = rle2mask(df.loc[i, "rle"], (df.loc[i, "img_height"], df.loc[i, "img_width"]))
        mask = Image.fromarray(mask)
        mask.save(f"inputs/Hacking the Human Body/{df.loc[i, 'id']}_mask.jpg")
