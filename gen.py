import sys
import bounce
import numpy as np

def main():
    go1 = bounce.GaussianObject(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1.0)
    go2 = bounce.GaussianObject(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 1.0)
    for _ in range(32):
        go1.push(go2)
        print(go2)
        go2.step(0.01)
        print(go2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user, exiting...")
        sys.exit(1)
