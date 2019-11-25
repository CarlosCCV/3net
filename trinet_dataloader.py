import tensorflow as tf
import read_parameters
import stereo_rectification
import os

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class TrinetDataloader(object):
    """trinet dataloader"""
    def __init__(self, data_path, filenames_file_train, filenames_file_validation, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.central_image_batch = None
        self.right_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file_train], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        left_image_path  = tf.string_join([self.data_path, split_line[0]])
        central_image_path = tf.string_join([self.data_path, split_line[1]])
        right_image_path = tf.string_join([self.data_path, split_line[2]])
        left_image_o  = self.read_image(left_image_path)
        central_image_o = self.read_image(central_image_path)
        right_image_o = self.read_image(right_image_path)

        if mode == 'train':

            self.left_image_val_batch  = None
            self.central_image_val_batch = None
            self.right_image_val_batch = None

            input_queue_val = tf.train.string_input_producer([filenames_file_validation], shuffle=False)
            line_reader_val = tf.TextLineReader()
            _, line_val = line_reader_val.read(input_queue_val)

            split_line_val = tf.string_split([line_val]).values

            left_image_val_path  = tf.string_join([self.data_path, split_line_val[0]])
            central_image_val_path = tf.string_join([self.data_path, split_line_val[1]])
            right_image_val_path = tf.string_join([self.data_path, split_line_val[2]])
            left_image_val_o  = self.read_image(left_image_val_path)
            central_image_val_o = self.read_image(central_image_val_path)
            right_image_val_o = self.read_image(right_image_val_path)

        # Calibration parameters of the cameras (DEBERIA CREAR UNA CLASE Y LEER ESTOS PARAMETROS DESDE LA FUNCION MAIN)
        self.intrinsics,self.extrinsics = read_parameters.read_calibration_parameters(self.data_path, filenames_file_train)

        A_left = self.intrinsics[:,:,0]
        A_center = self.intrinsics[:, :, 1]
        A_right = self.intrinsics[:, :, 2]

        extrinsics_left = self.extrinsics[:,:,0]
        extrinsics_center = self.extrinsics[:, :, 1]
        extrinsics_right = self.extrinsics[:, :, 2]


        # randomly flip images
        #do_flip = tf.random_uniform([], 0, 1)
        #left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
        #central_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)
        #right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o), lambda: right_image_o)

        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 2048
        capacity = min_after_dequeue + 4 * params.batch_size

        left_image = left_image_o
        central_image = central_image_o
        right_image = right_image_o


        left_image, cl_image = stereo_rectification.stereo_rectify(left_image, central_image, A_left, A_center, extrinsics_left, extrinsics_center, transformed_image_size=(self.params.height, self.params.width))
        cr_image, right_image = stereo_rectification.stereo_rectify(central_image, right_image, A_center, A_right, extrinsics_center, extrinsics_right, transformed_image_size=(self.params.height, self.params.width))


        # randomly augment images
        if mode == 'train':
            left_image_val = left_image_val_o
            central_image_val = central_image_val_o
            right_image_val = right_image_val_o
            left_image_val, cl_image_val = stereo_rectification.stereo_rectify(left_image_val, central_image_val, A_left,
                                                                             A_center, extrinsics_left,
                                                                             extrinsics_center,
                                                                             transformed_image_size=(
                                                                             self.params.height, self.params.width))
            cr_image_val, right_image_val = stereo_rectification.stereo_rectify(central_image_val, right_image_val, A_center,
                                                                              A_right, extrinsics_center,
                                                                              extrinsics_right,
                                                                              transformed_image_size=(
                                                                              self.params.height,
                                                                              self.params.width))
            # do_augment  = tf.random_uniform([], 0, 1)
            # left_image, cl_image, cr_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_images(left_image, cl_image, cr_image, right_image), lambda: (left_image, cl_image, cr_image, right_image))

            left_image_val.set_shape([None, None, 3])
            cl_image_val.set_shape([None, None, 3])
            cr_image_val.set_shape([None, None, 3])
            right_image_val.set_shape([None, None, 3])

            self.left_image_val_batch, self.cl_image_val_batch, self.cr_image_val_batch, self.right_image_val_batch = tf.train.shuffle_batch(
                [left_image_val, cl_image_val, cr_image_val, right_image_val],
                params.batch_size, capacity, min_after_dequeue, params.num_threads)

        left_image.set_shape( [None, None, 3])
        cl_image.set_shape([None, None, 3])
        cr_image.set_shape([None, None, 3])
        right_image.set_shape([None, None, 3])

        print(left_image.shape)


        self.left_image_batch, self.cl_image_batch, self.cr_image_batch, self.right_image_batch = tf.train.shuffle_batch([left_image, cl_image, cr_image, right_image],
                    params.batch_size, capacity, min_after_dequeue, params.num_threads)



        #elif mode == 'test':
        #    self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
        #    self.left_image_batch.set_shape( [2, None, None, 3])

        #    if self.params.do_stereo:
        #        self.right_image_batch = tf.stack([right_image_o,  tf.image.flip_left_right(right_image_o)],  0)
        #        self.right_image_batch.set_shape( [2, None, None, 3])

    def augment_images(self, left_image, cl_image, cr_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        cl_image_aug = cl_image ** random_gamma
        cr_image_aug = cr_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        cl_image_aug = cl_image_aug * random_brightness
        cr_image_aug = cr_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        cl_image_aug *= color_image
        cr_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        cl_image_aug = tf.clip_by_value(cl_image_aug, 0, 1)
        cr_image_aug = tf.clip_by_value(cr_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, cl_image_aug, cr_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        print("Call to read_image")
        print(image_path)
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        image  = tf.image.convert_image_dtype(image,  tf.float32)
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
    def reduce_image(self, image):
        image  = tf.image.convert_image_dtype(image,  tf.float32)
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image

    def create_mask(self, image):
        return image > 0