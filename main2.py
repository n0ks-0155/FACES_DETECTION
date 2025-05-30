import matplotlib.pyplot as plt
from glob import glob


def show_cropped_images():
    cropped_images = glob('./faces/cut/*.jpg')

    plt.figure(figsize=(12, 8))
    plt.suptitle('Training Images')

    for i, img_path in enumerate(cropped_images[:9], 1):
        ax = plt.subplot(3, 3, i)
        img_title = 'face: ' + img_path[12:-4]
        plt.title(img_title, fontsize='medium')
        image = plt.imread(img_path)
        plt.imshow(image, cmap=plt.cm.binary)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_cropped_images()