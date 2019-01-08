import time
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
from random_erasing import random_erasing

# numpy: 0.1695 sec, TF: 0.1696 sec

def random_erase_np(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height
    for attempt in range(100):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1/r1)

        h = int(np.round(np.sqrt(target_area * aspect_ratio)))
        w = int(np.round(np.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:
            x1 = np.random.randint(0, height - h)
            y1 = np.random.randint(0, width - w)
            img[x1:x1+h, y1:y1+w, :] = np.random.uniform(0, 255, (h, w, channel))
            return img

    return img

def random_erase_np_v2(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height

    erase_area_low_bound = np.round( np.sqrt(sl * area * r1) ).astype(np.int)
    erase_area_up_bound = np.round( np.sqrt((sh * area) / r1) ).astype(np.int)
    if erase_area_up_bound < height:
        h_upper_bound = erase_area_up_bound
    else:
        h_upper_bound = height
    if erase_area_up_bound < width:
        w_upper_bound = erase_area_up_bound
    else:
        w_upper_bound = width

    h = np.random.randint(erase_area_low_bound, h_upper_bound)
    w = np.random.randint(erase_area_low_bound, w_upper_bound)

    x1 = np.random.randint(0, height+1 - h)
    y1 = np.random.randint(0, width+1 - w)

    x1 = np.random.randint(0, height - h)
    y1 = np.random.randint(0, width - w)
    img[x1:x1+h, y1:y1+w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.uint8)
    return img

if __name__ == '__main__':
    sess = tf.Session()
    raw_jpeg = tf.read_file("./data/cat.jpg")
    image = tf.image.decode_jpeg(raw_jpeg, channels=3)

    # cause decode_jpeg output type is undefined until sess.run(), so must close validate init value shape
    image_var = tf.Variable(image, validate_shape=False)
    init_op = tf.variables_initializer(var_list=[image_var])

    for _ in range(5):
        start = time.time()
        img1 = tf.py_func(random_erase_np_v2, [image], tf.uint8)
        imsave("random_erasing_np.jpg", sess.run(img1))
        print("Numpy version: {}".format(time.time()-start))

        sess.run(init_op)

        start = time.time()
        img = random_erasing(image_var)
        imsave("random_erasing.jpg", sess.run(img))
        print("TF version: {}".format(time.time()-start))

        print("")