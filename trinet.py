#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from layers import *
from utils import *
from collections import namedtuple
import stereo_rectification

trinet_parameters = namedtuple('parameters',
                               'encoder, '
                               'height, width, '
                               'batch_size, '
                               'num_threads, '
                               'num_epochs, '
                               'alpha_image_loss, '
                               'disp_gradient_loss_weight, '
                               'lr_loss_weight, '
                               'full_summary')

class trinet(object):

    def __init__(self,params, mode, left, cl, cr, right, intrinsics, extrinsics, reuse_variables=None, model_index=0, net='vgg'):
        self.params = params
        self.mode = mode
        self.model_collection = ['model_0']
        self.left = left
        self.right = right
        self.cl = cl
        self.cr = cr
        self.reuse_variables = reuse_variables
        self.model_index = model_index

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.build_model(net)
        self.build_outputs()
        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    # Build model
    def build_model(self,net): 
        with tf.variable_scope('model', reuse=self.reuse_variables) as scope:
          self.left_pyramid = self.scale_pyramid(self.left, 4)
          # if self.mode == 'train':
          self.right_pyramid = self.scale_pyramid(self.right, 4)
          self.cl_pyramid = self.scale_pyramid(self.cl, 4)
          self.cr_pyramid = self.scale_pyramid(self.cr, 4)


          with tf.variable_scope('encoder'):
            features_cr = self.build_encoder(self.cr,model_name=net)
            features_cl = self.build_encoder(self.cl, model_name=net)
          with tf.variable_scope('encoder-C2R'):
            self.disp_c2r = self.build_decoder(features_cr,model_name=net)
          with tf.variable_scope('encoder-C2L'):
            self.disp_c2l = self.build_decoder(features_cl,model_name=net)
      
    # Build shared encoder
    def build_encoder(self, model_input, model_name='vgg'):

        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
          if model_name == 'vgg':
            conv1 = conv_block(model_input,  32, 7) # H/2
            conv2 = conv_block(conv1,             64, 5) # H/4
            conv3 = conv_block(conv2,            128, 3) # H/8
            conv4 = conv_block(conv3,            256, 3) # H/16
            conv5 = conv_block(conv4,            512, 3) # H/32
            conv6 = conv_block(conv5,            512, 3) # H/64
            conv7 = conv_block(conv6,            512, 3) # H/128    
            return conv7, conv1, conv2, conv3, conv4, conv5, conv6

          elif model_name == 'resnet50':
            conv1 = conv(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D
            return conv5, conv1, pool1, conv2, conv3, conv4      

    def build_decoder(self, skip, model_name='vgg'):

        with tf.variable_scope('decoder'):
          if model_name == 'vgg':           
            upconv7 = upconv(skip[0],  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip[6]], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip[5]], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip[4]], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip[3]], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip[2], udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip[1], udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)

          elif model_name == 'resnet50':            
            upconv6 = upconv(skip[0],   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip[5]], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip[4]], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip[3]], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip[2], udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip[1], udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)

          return disp1, disp2, disp3, disp4    
    def build_outputs(self):
        #self.disparity_cr = self.disp_cr[0][0,:,:,0]
        #self.disparity_cl = self.disp_cl[0][0,:,:,0]
        #self.warp_left = generate_image_left(self.placeholders['im0'], self.disparity_cl)[0]
        #self.warp_right = generate_image_right(self.placeholders['im0'], self.disparity_cr)[0]

        # STORE DISPARITIES
        with tf.variable_scope('disparities'):

            self.disp_lc = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_c2l]
            self.disp_cl = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_c2l]

            self.disp_cr = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_c2r]
            self.disp_rc = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_c2r]

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.cl_pyramid[i], self.disp_lc[i]) for i in range(4)]
            self.cl_est = [self.generate_image_right(self.left_pyramid[i], self.disp_cl[i]) for i in range(4)]

            self.cr_est = [self.generate_image_left(self.right_pyramid[i], self.disp_cr[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.cr_pyramid[i], self.disp_rc[i]) for i in range(4)]

        # UNRECTIFY CENTRAL IMAGES
            print('LEEFT EST')
            print(type(self.left_est))
            self.cl_est_unrect = [None] * 4
            self.cr_est_unrect = [None] * 4

            for i in range(4):
                _,self.cl_est_unrect[i] = stereo_rectification.unrectify(self.left_est[i], self.cl_est[i], self.intrinsics[:,:,0], self.intrinsics[:,:,1], self.extrinsics[:,:,0],self.extrinsics[:,:,1], transformed_image_size=(self.params.height,self.params.width))
                self.cr_est_unrect[i], _ = stereo_rectification.unrectify(self.cr_est[i], self.right_est[i], self.intrinsics[:, :, 1], self.intrinsics[:, :, 2], self.extrinsics[:, :, 1], self.extrinsics[:, :, 2], transformed_image_size=(self.params.height, self.params.width))


        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.cl_to_lc_disp = [self.generate_image_left(self.disp_cl[i], self.disp_lc[i]) for i in range(4)]
            self.lc_to_cl_disp = [self.generate_image_right(self.disp_lc[i], self.disp_cl[i]) for i in range(4)]

            self.rc_to_cr_disp = [self.generate_image_left(self.disp_rc[i], self.disp_cr[i]) for i in range(4)]
            self.cr_to_rc_disp = [self.generate_image_right(self.disp_cr[i], self.disp_rc[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_lc_smoothness  = self.get_disparity_smoothness(self.disp_lc,  self.left_pyramid)
            self.disp_cl_smoothness = self.get_disparity_smoothness(self.disp_cl, self.cl_pyramid)

            self.disp_cr_smoothness = self.get_disparity_smoothness(self.disp_cr, self.cr_pyramid)
            self.disp_rc_smoothness = self.get_disparity_smoothness(self.disp_rc, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]

            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            self.l1_cl = [tf.abs(self.cl_est[i] - self.cl_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_cl = [tf.reduce_mean(l) for l in self.l1_cl]

            self.l1_cr = [tf.abs(self.cr_est[i] - self.cr_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_cr = [tf.reduce_mean(l) for l in self.l1_cr]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]

            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            self.ssim_cl = [self.SSIM(self.cl_est[i], self.cl_pyramid[i]) for i in range(4)]
            self.ssim_loss_cl = [tf.reduce_mean(s) for s in self.ssim_cl]

            self.ssim_cr = [self.SSIM(self.cr_est[i], self.cr_pyramid[i]) for i in range(4)]
            self.ssim_loss_cr = [tf.reduce_mean(s) for s in self.ssim_cr]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss_cl = [self.params.alpha_image_loss * self.ssim_loss_cl[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_cl[i] for i in range(4)]
            self.image_loss_cr = [self.params.alpha_image_loss * self.ssim_loss_cr[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_cr[i] for i in range(4)]

            self.image_loss = tf.add_n(self.image_loss_left +  self.image_loss_cl + self.image_loss_right + self.image_loss_cr)

            self.image_loss_L = tf.add_n(self.image_loss_left +  self.image_loss_cl)
            self.image_loss_R = tf.add_n(self.image_loss_right + self.image_loss_cr)


            # DISPARITY SMOOTHNESS
            self.disp_lc_loss = [tf.reduce_mean(tf.abs(self.disp_lc_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_cl_loss = [tf.reduce_mean(tf.abs(self.disp_cl_smoothness[i])) / 2 ** i for i in range(4)]

            self.disp_rc_loss = [tf.reduce_mean(tf.abs(self.disp_rc_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_cr_loss = [tf.reduce_mean(tf.abs(self.disp_cr_smoothness[i])) / 2 ** i for i in range(4)]

            self.disp_gradient_loss = tf.add_n(self.disp_lc_loss + self.disp_cl_loss + self.disp_rc_loss + self.disp_cr_loss)

            self.disp_gradient_loss_L = tf.add_n(self.disp_lc_loss + self.disp_cl_loss)
            self.disp_gradient_loss_R = tf.add_n(self.disp_rc_loss + self.disp_cr_loss)


            # LR CONSISTENCY
            self.lr_lc_loss = [tf.reduce_mean(tf.abs(self.cl_to_lc_disp[i] - self.disp_lc[i])) for i in range(4)]
            self.lr_cl_loss = [tf.reduce_mean(tf.abs(self.lc_to_cl_disp[i] - self.disp_cl[i])) for i in range(4)]

            self.lr_rc_loss = [tf.reduce_mean(tf.abs(self.cr_to_rc_disp[i] - self.disp_rc[i])) for i in range(4)]
            self.lr_cr_loss = [tf.reduce_mean(tf.abs(self.rc_to_cr_disp[i] - self.disp_cr[i])) for i in range(4)]


            self.lr_loss = tf.add_n(self.lr_lc_loss + self.lr_cl_loss + self.lr_rc_loss + self.lr_cr_loss)

            self.lr_loss_L = tf.add_n(self.lr_lc_loss + self.lr_cl_loss)
            self.lr_loss_R = tf.add_n(self.lr_rc_loss + self.lr_cr_loss)

            # CENTRAL RECONSTRUCTION CONSISTENCY
            self.central_reconstruction_dif = [tf.reduce_mean(tf.abs(self.cl_est_unrect[i] - self.cr_est_unrect[i])) for i in range(4)]
            self.central_reconstruction_loss = tf.add_n(self.central_reconstruction_dif)

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss + self.central_reconstruction_loss

            self.total_loss_L =  self.image_loss_L + self.params.disp_gradient_loss_weight * self.disp_gradient_loss_L + self.params.lr_loss_weight * self.lr_loss_L
            self.total_loss_R = self.image_loss_R + self.params.disp_gradient_loss_weight * self.disp_gradient_loss_R + self.params.lr_loss_weight * self.lr_loss_R

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_cl[i] + self.ssim_loss_right[i] + self.ssim_loss_cr[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_cl[i] + self.l1_reconstruction_loss_right[i] + self.l1_reconstruction_loss_cr[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_cl[i] + self.image_loss_right[i] + self.image_loss_cr[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_lc_loss[i] + self.disp_cl_loss[i] + self.disp_rc_loss[i] + self.disp_cr_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_lc_loss[i] + self.lr_cl_loss[i] + self.lr_rc_loss[i] + self.lr_cr_loss[i], collections=self.model_collection)
                tf.summary.scalar('total_loss_L', self.total_loss_L, collections= self.model_collection)
                tf.summary.scalar('total_loss_R', self.total_loss_R, collections=self.model_collection)
                tf.summary.scalar('central_reconstruction_loss', self.central_reconstruction_loss, collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_lc[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_cl_est_' + str(i), self.disp_cl[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_rc[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_cr_est_' + str(i), self.disp_cr[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('left_pyramid_' + str(i), self.left_pyramid[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cr_pyramid_' + str(i), self.cr_pyramid[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cl_pyramid_' + str(i), self.cl_pyramid[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('right_pyramid_' + str(i), self.right_pyramid[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cr_est_' + str(i), self.cr_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cl_est_' + str(i), self.cl_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cr_est_unrect_' + str(i), self.cr_est_unrect[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cl_est_unrect_' + str(i), self.cl_est_unrect[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('cl', self.cl, max_outputs=4, collections=self.model_collection)
                tf.summary.image('cr', self.cr, max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    #tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('cl_est_' + str(i), self.cl_est[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('cr_est_' + str(i), self.cr_est[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('ssim_cl_' + str(i), self.ssim_cl[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('ssim_cr_' + str(i), self.ssim_cr[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)
                    #tf.summary.image('l1_cl_' + str(i), self.l1_cl[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_cr_' + str(i), self.l1_cr[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)
                tf.summary.image('cl', self.cl, max_outputs=4, collections=self.model_collection)
                tf.summary.image('cr', self.cr, max_outputs=4, collections=self.model_collection)