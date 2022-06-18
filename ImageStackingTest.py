import ImageStacking
import os
def test_shape():
    os.chdir('Image Segmentation/train')
    mri_list, mri_names = ImageStacking.main(os.getcwd())

    assert len(mri_list[0].shape) == 3
    print(mri_list[0], mri_list[0].shape)


def main():
    test_shape()
if __name__ == '__main__':
    main()