from __future__ import absolute_import, division, print_function
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import argparse
import tensorflow as tf

import time
import numpy as np

from trinet import *
from trinet_dataloader import *
from average_gradients import *
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description = "Trinet Tensorflow implementation.")

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')

args = parser.parse_args()

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values) # Done for learning rate decay

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = TrinetDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left = dataloader.left_image_batch
        cl = dataloader.cl_image_batch
        cr = dataloader.cr_image_batch
        right = dataloader.right_image_batch

        #split for each gpu
        left_splits = tf.split(left, args.num_gpus, 0)
        cl_splits = tf.split(cl, args.num_gpus, 0)
        cr_splits = tf.split(cr, args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        #tower_grads_L = []
        #tower_losses_L = []
        #tower_grads_R = []
        #tower_losses_R = []
        tower_losses = []
        tower_grads = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    model = trinet(params, args.mode, left_splits[i], cl_splits[i], cr_splits[i], right_splits[i], dataloader.intrinsics, dataloader.extrinsics, reuse_variables, i)

                    #loss_L = model.total_loss_L
                    #loss_R = model.total_loss_R
                    loss = model.total_loss

                    #tower_losses_L.append(loss_L)
                    #tower_losses_R.append(loss_R)
                    tower_losses.append(loss)

                    reuse_variables = True

                    #grads_L = opt_step.compute_gradients(loss_L)#, [first_train_vars, second_train_vars])
                    #grads_R = opt_step.compute_gradients(loss_R)  # , [first_train_vars, second_train_vars])
                    grads = opt_step.compute_gradients(loss)

                    #tower_grads_L.append(grads_L)
                    #tower_grads_R.append(grads_R)
                    tower_grads.append(grads)

        #grads_L = average_gradients(tower_grads_L)
        #grads_R = average_gradients(tower_grads_R)


        #apply_gradient_op_L = opt_step.apply_gradients(grads_L, global_step=global_step)
        #apply_gradient_op_R = opt_step.apply_gradients(grads_R, global_step=global_step)
        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        #total_loss_L = tf.reduce_mean(tower_losses_L)
        #total_loss_R = tf.reduce_mean(tower_losses_R)
        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        #tf.summary.scalar('total_loss_L', total_loss_L, ['model_0'])
        #tf.summary.scalar('total_loss_R', total_loss_R, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')


        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()

        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            #_, loss_value_L = sess.run([apply_gradient_op_L, total_loss_L])
            #_, loss_value_R = sess.run([apply_gradient_op_R, total_loss_R])
            _, loss_value = sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 1000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = TrinetDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left = dataloader.left_image_batch
    central = dataloader.central_image_batch
    right = dataloader.right_image_batch

    model = trinet(params, args.mode, left, central, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(13):
        print('Testing batch {}'.format(step))
        fetches = [model.disp_cl, model.disp_cr, model.disp_c2l]
        [disp_cl, disp_cr, disp_c2l] = sess.run(fetches)
        print(disp_c2l[1].shape)
        plt.imshow(disp_c2l[0][0,:,:,0])
        plt.show()
        plt.pause(2)
        plt.imshow(disp_c2l[0][0, :, :, 1])
        plt.show()
        plt.pause(2)
        disparities[step] = disp_cl[0].squeeze()

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)

    print('done.')


def main(_):

    params = trinet_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        #sys.exit('Test function not implemented yet . . .')
        test(params)

if __name__ == '__main__':
    tf.app.run()

