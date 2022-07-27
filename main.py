import TransformerUNet
import Train_model

import CropVolumes

import DataLoader

DATA_URL = ""
LABEL_URL = ""


def clean():
    pass


def fit_model():
    pass


def main():
    use = input("Clean Data, Fit Model or Both. (1,2, or 3)")
    if use == 1:
        clean()

    elif use == 2:
        fit_model()

    else:
        clean()
        fit_model()


if __name__ == "__main__":
    main()
