from pathlib import Path
import os
import cv2
import re
from tqdm import tqdm


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    img_path = Path("./mouseRunningData/barObstacleScaling1/").resolve()
    TOP_IMG_HEIGHT = 168
    BOT_IMG_HEIGHT = 238

    images = os.listdir(img_path)
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    images = sorted_nicely(images)
    save_dir = Path("./mouseRunningData/")
    top_dir = save_dir / Path("camera1Images/")
    top_dir.mkdir(parents=True, exist_ok=True)
    bot_dir = save_dir / Path("camera2Images/")
    bot_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(images):
        if "png" not in i:
            continue
        img = cv2.imread(str(img_path / i), 0)
        top_img = img[:TOP_IMG_HEIGHT, :]
        bot_img = img[TOP_IMG_HEIGHT:, :]
        cv2.imwrite(str(top_dir / i), top_img)
        cv2.imwrite(str(bot_dir / i), bot_img)