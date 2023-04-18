from sklearn.cluster import KMeans
import segmentation_models as sm
from sklearn.cluster import KMeans
import albumentations as A
import matplotlib.pyplot as plt
from .dataset import Dataset, Dataloder
import numpy as np
import keras as keras
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# %matplotlib inline


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


BACKBONE = 'densenet201'
BATCH_SIZE = 6
CLASSES = ['lobe-1', 'lobe-2', 'lobe-3', 'lobe-4', 'lobe-5']
LR = 0.0001
preprocess_input = sm.get_preprocessing(BACKBONE)
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
activation = 'sigmoid' if n_classes == 1 else 'softmax'
keras.backend.clear_session()
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss(
    class_weights=np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.005]))
focal_loss = sm.losses.BinaryFocalLoss(
) if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)
model.load_weights('static/weights.h5')


def predict(image):
    global CLASSES, preprocess_input, BATCH_SIZE, n_classes, model
    test_dataset = Dataset(
        image=image,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
    )
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
    output = model.predict(test_dataloader)
    # with open('pred.npy', 'wb') as f:
    #     np.save(f, output)
    # print(output)
    return output


def visualize(image):
    """PLot images in one row."""

    plt.figure(figsize=(16, 5))
    #plt.subplot(1, n, i + 1)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)
    plt.show()


def visualize_one_line(images):
    """PLot images in one row."""
    n = len(images)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 3, 1)

    plt.imshow(images[0])
    plt.axis('off')
    plt.title("original")

    fig.add_subplot(1, 2, 2)
    plt.imshow(images[1])
    plt.axis('off')
    plt.title("proc-1")

    fig.add_subplot(1, 3, 3)
    plt.imshow(images[2])
    plt.axis('off')
    plt.title("proc-2")


def blood_vessels_removal(image, acc_mask):
    fimg = image
    facc_mask = acc_mask
    facc_mask = facc_mask/255
    facc_mask[np.where(facc_mask > 0)] = 1
    mimg = np.multiply(fimg, facc_mask)

    mimg_og = mimg.copy()

    # Getting the kernel to be used in Top-Hat
    filterSize = (9, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)

    # Reading the image named 'input.jpg'
    input_image = mimg

    # Applying the Top-Hat operation
    tophat_img = cv2.morphologyEx(input_image,
                                  cv2.MORPH_TOPHAT,
                                  kernel)
    # normalize vals between 0 and 1
    tophat_img = np.divide(tophat_img, np.max(tophat_img))
    inv_th_img = np.ones((512, 512, 3))
    # thresholding operation
    inv_th_img[np.where(np.logical_and(tophat_img > 0.2, tophat_img <= 1))] = 0
    nimg = np.divide(mimg, 255)  # normalize vals between 0 and 1

    tophat_thres_img = tophat_img/255
#     x = np.zeros((512,512,3))
#     x[np.where(np.logical_not(np.logical_and(tophat_thres_img > 0.10, tophat_thres_img <= 1)))] = 1
#     tmpimg = np.multiply(nimg,x)

    timg = np.divide(mimg, np.max(mimg))
    tmpimg = np.multiply(timg, inv_th_img)  # large vessels removal

    # small vessels removal
    gaussian_img = cv2.GaussianBlur(mimg, (3, 3), cv2.BORDER_DEFAULT)
    dimg = np.divide(tophat_img, gaussian_img+np.finfo(float).eps)
    opening = cv2.morphologyEx(dimg, cv2.MORPH_OPEN, (3, 3))
    invimg = opening.copy()
    invimg = np.divide(invimg, np.max(invimg))
    inv_th_img = np.ones((512, 512, 3))
    inv_th_img[np.where(invimg == 0)] = 0
    inv_th_img[np.where(np.logical_and(invimg > 0, invimg <= 0.02))] = 0
    ftmpimg = np.multiply(tmpimg, inv_th_img)

    fpremoved_img = ftmpimg.copy()

    #tmpimg[np.where(tmpimg > 0.80)] = 0
    tmp = ftmpimg[:, :, 0]
    flatimg = np.ndarray.flatten(tmp)
    nz_idx = np.where(flatimg > 0)
    nz_flatimg = flatimg[np.where(flatimg > 0)]
    nimg = nz_flatimg
    nimg = nimg.reshape((nimg.shape[0], 1))

    clustering = KMeans(n_clusters=2, random_state=0).fit(nimg)
    v1 = nimg[np.where(clustering.labels_ == 0)]
    v2 = nimg[np.where(clustering.labels_ == 1)]
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)
    mean_arr = np.array([mean_v1, mean_v2])
    max_mean_idx = np.argmax(mean_arr)

    if max_mean_idx == 0:
        new_labels = np.where((clustering.labels_ == 0) | (
            clustering.labels_ == 1), clustering.labels_ ^ 1, clustering.labels_)
    else:
        new_labels = clustering.labels_

    dispimg = np.zeros(flatimg.shape)
    dispimg = np.ndarray.flatten(dispimg)
    dispimg[nz_idx] = np.array(new_labels)
    dispimg = dispimg.reshape((512, 512))

    # just for sake of displaying
    c1 = dispimg.copy()
    c2 = dispimg.copy()
    fdispimg = np.dstack([dispimg, c1, c2])

    return fpremoved_img, fdispimg


def ggo_detect(image, acc_mask):
    arr = list()
    for i in range(6):
        fimg = image
        facc_mask = acc_mask[:, :, :, [i]][0]
        facc_mask = (facc_mask*255).astype(np.uint8)
        facc_mask = facc_mask/255
        facc_mask[np.where(facc_mask > 0)] = 1
        mimg = np.multiply(fimg, facc_mask)
        mimg_og = mimg.copy()
        filterSize = (9, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        filterSize)
        input_image = mimg
        tophat_img = cv2.morphologyEx(input_image,
                                    cv2.MORPH_TOPHAT,
                                    kernel)
        tophat_img = np.divide(tophat_img, np.max(tophat_img))
        inv_th_img = np.ones((512, 512, 3))
        inv_th_img[np.where(np.logical_and(tophat_img > 0.2, tophat_img <= 1))] = 0
        nimg = np.divide(mimg, 255)
        tophat_thres_img = tophat_img/255
        timg = np.divide(mimg, np.max(mimg))
        tmpimg = np.multiply(timg, inv_th_img)
        gaussian_img = cv2.GaussianBlur(mimg, (3, 3), cv2.BORDER_DEFAULT)
        dimg = np.divide(tophat_img, gaussian_img+np.finfo(float).eps)
        opening = cv2.morphologyEx(dimg, cv2.MORPH_OPEN, (3, 3))
        invimg = opening.copy()
        invimg = np.divide(invimg, np.max(invimg))
        inv_th_img = np.ones((512, 512, 3))
        inv_th_img[np.where(invimg == 0)] = 0
        inv_th_img[np.where(np.logical_and(invimg > 0, invimg <= 0.02))] = 0
        ftmpimg = np.multiply(tmpimg, inv_th_img)

        fpremoved_img = ftmpimg.copy()
        tmp = ftmpimg[:, :, 0]
        flatimg = np.ndarray.flatten(tmp)
        nz_idx = np.where(flatimg > 0)
        nz_flatimg = flatimg[np.where(flatimg > 0)]
        nimg = nz_flatimg
        nimg = nimg.reshape((nimg.shape[0], 1))

        clustering = KMeans(n_clusters=2, random_state=0).fit(nimg)
        v1 = nimg[np.where(clustering.labels_ == 0)]
        v2 = nimg[np.where(clustering.labels_ == 1)]
        mean_v1 = np.mean(v1)
        mean_v2 = np.mean(v2)
        mean_arr = np.array([mean_v1, mean_v2])
        max_mean_idx = np.argmax(mean_arr)

        if max_mean_idx == 0:
            new_labels = np.where((clustering.labels_ == 0) | (
                clustering.labels_ == 1), clustering.labels_ ^ 1, clustering.labels_)
        else:
            new_labels = clustering.labels_

        dispimg = np.zeros(flatimg.shape)
        dispimg = np.ndarray.flatten(dispimg)
        dispimg[nz_idx] = np.array(new_labels)
        dispimg = dispimg.reshape((512, 512))
        c1 = dispimg.copy()
        c2 = dispimg.copy()
        fdispimg = np.dstack([dispimg, c1, c2])

        fpr = fpremoved_img
        f1 = fdispimg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (2, 2))
        f2 = cv2.morphologyEx(f1, cv2.MORPH_OPEN, kernel)
        arr.append([fpr, f2])
    return arr
