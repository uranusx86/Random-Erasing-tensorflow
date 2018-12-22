import tensorflow as tf

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)

    target_area = tf.random.uniform([100], sl, sh) * area
    aspect_ratio = tf.random.uniform([100], r1, 1/r1)

    h = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
    w = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

    w_cond = tf.less(w, width)
    h_cond = tf.less(h, height)
    cond = tf.logical_and(w_cond, h_cond)
    cond_true_idx = tf.cast(cond, tf.int32)
    first_true_idx = tf.argmax(cond_true_idx)

    x1 = tf.random.uniform([], 0, height - h[first_true_idx], tf.int32)
    y1 = tf.random.uniform([], 0, width - w[first_true_idx], tf.int32)

    erase_area = tf.cast(tf.random.uniform([h[first_true_idx], w[first_true_idx], channel], 0, 255, tf.int32), tf.uint8)

    erasing_img = img[x1:x1+h[first_true_idx], y1:y1+w[first_true_idx], :].assign(erase_area)

    can_find_idx = tf.cond(tf.equal(tf.reduce_sum(tf.cast(cond, tf.int32)), 0), lambda: img, lambda: erasing_img)  # if cant find True, return original img

    return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: can_find_idx)