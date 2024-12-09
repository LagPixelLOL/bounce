import os
import sys
import math
import tqdm
import bounce
import datetime
import argparse
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a bouncing animation.")
    parser.add_argument("-f", "--foreground", default="ffffff", help="Hex color code for foreground")
    parser.add_argument("-b", "--background", default="000000", help="Hex color code for background")
    parser.add_argument("-n", "--n-objects", type=int, default=8, help="Number of objects on the plane")
    parser.add_argument("-v", "--velocity", type=float, default=1.0, help="Initial magnitude of velocity for the objects")
    parser.add_argument("-d", "--std", type=float, default=1.0, help="Standard deviation for the objects")
    parser.add_argument("-S", "--seed", type=int, default=None, help="The seed used to spawn the objects")
    parser.add_argument("-W", "--width", type=int, default=256, help="Width of the output")
    parser.add_argument("-H", "--height", type=int, default=256, help="Height of the output")
    parser.add_argument("-r", "--resolution", type=float, default=0.05, help="Distance on the plane between each pixel")
    parser.add_argument("-q", "--fps", type=int, default=30, help="FPS for the output")
    parser.add_argument("-t", "--time-step", type=float, default=0.05, help="Time resolution for each step")
    parser.add_argument("-s", "--total-steps", type=int, default=512, help="Total steps to run")
    parser.add_argument("-p", "--render-steps", type=int, default=4, help="Render every n steps")
    args = parser.parse_args()

    def hex_to_rgb(hex_str):
        try:
            base_10 = np.uint32(int(hex_str.strip("#-\n "), 16))
        except ValueError as e:
            print("Error while parsing color code hex to base 10 integer:", e)
            sys.exit(1)
        if base_10 < 0 or base_10 > 16777215:
            print("Color code must be between #000000 and #FFFFFF, instead got", f"#{base_10:0>6x}!".upper())
            sys.exit(1)
        return np.array([base_10 >> 16, base_10 >> 8, base_10], dtype=np.uint8)

    args.foreground = hex_to_rgb(args.foreground)
    args.background = hex_to_rgb(args.background)
    if args.n_objects < 0:
        print("Number of objects on the plane must be non-negative!")
        sys.exit(1)
    if args.velocity < 0:
        print("Initial magnitude of velocity must be non-negative!")
        sys.exit(1)
    if args.std <= 0:
        print("Standard deviation must be positive!")
        sys.exit(1)
    if args.width < 1:
        print("Width of the output must be positive!")
        sys.exit(1)
    if args.height < 1:
        print("Height of the output must be positive!")
        sys.exit(1)
    if args.resolution <= 0:
        print("Resolution must be positive!")
        sys.exit(1)
    if args.fps < 1:
        print("FPS must be positive!")
        sys.exit(1)
    if args.time_step <= 0:
        print("Time resolution must be positive!")
        sys.exit(1)
    if args.total_steps < 1:
        print("Total steps must be positive!")
        sys.exit(1)
    if args.render_steps < 1:
        print("Render steps must be positive!")
        sys.exit(1)
    if args.render_steps >= args.total_steps:
        print("Render steps must be less than total steps!")
        sys.exit(1)
    return args

def main():
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    positive_x_bound = args.width / 2 * args.resolution
    negative_x_bound = -positive_x_bound
    positive_y_bound = args.height / 2 * args.resolution
    negative_y_bound = -positive_y_bound

    gaussian_objects = []
    for _ in range(args.n_objects):
        pos2d = np.array([generator.uniform(negative_x_bound, positive_x_bound), generator.uniform(negative_y_bound, positive_y_bound)])
        theta = generator.uniform(0, math.tau)
        vel2d = np.array([np.cos(theta), np.sin(theta)]) * args.velocity
        gaussian_objects.append(bounce.GaussianObject(pos2d, vel2d, args.std, args.foreground))
    box = bounce.Box(gaussian_objects, np.array([negative_x_bound, positive_y_bound]), np.array([positive_x_bound, negative_y_bound]), args.background)

    images = []
    for i in tqdm.tqdm(range(args.total_steps), desc="Running"):
        box.step(args.time_step)
        if i % args.render_steps == 0:
            images.append(Image.fromarray(box.render(args.resolution)))
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    images[0].save(
        os.path.join(outputs_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.gif")),
        save_all=True,
        append_images=images[1:],
        duration=1000 // args.fps,
        loop=0,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
