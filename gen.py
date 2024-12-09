import sys
import tqdm
import bounce
import numpy as np
from PIL import Image

def main():
    go1 = bounce.GaussianObject(np.array([5.0, 5.0]), np.array([-1.1, -0.9]), 1.0, np.array([101, 137, 164], dtype=np.int16))
    go2 = bounce.GaussianObject(np.array([-5.0, -5.0]), np.array([1.1, 0.9]), 1.0, np.array([101, 137, 164], dtype=np.int16))
    box = bounce.Box([go1, go2], np.array([-10.0, 10.0]), np.array([10.0, -10.0]), np.array([49, 60, 83], dtype=np.uint8))
    images = []
    for i in tqdm.tqdm(range(512), desc="Running"):
        box.step(0.05)
        if i % 4 == 0:
            images.append(Image.fromarray(box.render(0.05)))
    images[0].save(
        "ztmp.gif",
        save_all=True,
        append_images=images[1:],
        duration=1000 // 60,
        loop=0,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
