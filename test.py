import glob, os, sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise Exception("make sure you supply output_path, data_path and gpu num as sys arguments")

    output_path = sys.argv[1]

    ckpts = sorted([f for f in glob.glob(output_path + "/*/*.pth")])
    configs = sorted([f for f in glob.glob(output_path + "/*/*.json")])
    print(ckpts)
    print(configs)

