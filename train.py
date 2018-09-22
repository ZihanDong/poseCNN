# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from GoogLeNet import GoogLeNet
from PoseCNN import PoseCNN
import cv2
from tqdm import tqdm


batch_size = 100
max_iterations = 10000
# Set this path to your dataset directory
data_idx = '_data_mvs.txt'
# Set this path to your working space directory
outputFile = "C:/Users/Zihan/OneDrive/Pro/poseCNN/PoseCNN.ckpt"

class datasource(object):
    def __init__(self, images1, images2, poses, idx, max_size):
        self.images1 = images1
        self.images2 = images2
        self.poses = poses
        self.max_size = max_size
        self.idx = idx
        self.pos = 0

def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)
    cropped_img = img[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    return cropped_img

def preprocess(images):
    images_out = [] #final result
    #Resize and crop and compute mean!
    images_cropped = []
    for i in range(len(images)):
        X = cv2.imread(images[i])
        X = cv2.resize(X, (320, 240))
        X = centeredCrop(X, 224)
        images_cropped.append(X)
    #compute images mean
    N = 0
    mean = np.zeros((1, 3, 224, 224))
    for X in images_cropped:
        X = np.transpose(X,(2,0,1))
        mean[0][0] += X[0,:,:]
        mean[0][1] += X[1,:,:]
        mean[0][2] += X[2,:,:]
        N += 1
    mean[0] /= N
    #Subtract mean from all images
    for X in images_cropped:
        X = np.transpose(X,(2,0,1))
        X = X - mean
        X = np.squeeze(X)
        X = np.transpose(X, (1,2,0))
        images_out.append(X)
    return images_out

def get_data(mode = 'train', sub_sample = 'Cleaned'):
    poses = []
    images1 = []
    images2 = []

    with open('data/'+ mode + data_idx) as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            imgFiledId1, imgFiledId2, categoryId, x, y, z, q1, q2, q3, q4 = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            q1 = float(q1)
            q2 = float(q2)
            q3 = float(q3)
            q4 = float(q4)
            poses.append((x,y,z,q1,q2,q3,q4))
            
            imgFiledId1 = '0'+imgFiledId1 if len(imgFiledId1)==1 else imgFiledId1
            imgFiledId2 = '0'+imgFiledId2 if len(imgFiledId2)==1 else imgFiledId2
            images1.append('D:/Dataset/' + sub_sample + '/scan'+categoryId+'/'\
                           +'clean_0' + imgFiledId1 + '_max.png')
            images2.append('D:/Dataset/' + sub_sample + '/scan'+categoryId+'/'\
                           +'clean_0' + imgFiledId2 + '_max.png')
    max_size = len(poses)
    indices = list(range(max_size))
    random.shuffle(indices)
    return datasource(images1, images2, poses, indices, max_size)

def gen_data_batch(source):
    image1_batch = []
    image2_batch = []
    pose_x_batch = []
    pose_q_batch = []
    for i in range(batch_size):
        pos = i + source.pos
        pose_x = source.poses[i][0:3]
        pose_q = source.poses[i][3:7]
        image1_batch.append(source.images1[pos])
        image2_batch.append(source.images2[pos])
        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)
    image1_batch = preprocess(image1_batch)
    image2_batch = preprocess(image2_batch)
    source.pos += i
    if source.pos + i > source.max_size:
        source.pos = 0
    return np.array(image1_batch), np.array(image2_batch), np.array(pose_x_batch), np.array(pose_q_batch)


def main():
    # Create 2 separate graphs
    graph_GoogLeNet = tf.Graph()
    graph_PoseCNN = tf.Graph()
    #outputFile_GoogLeNet = "GoogLeNet.ckpt"
    datasource = get_data()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #Build GoogLeNet
    with graph_GoogLeNet.as_default():
        #place holder for input
        images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name = 'input')
        
        # define network
        googLeNet = GoogLeNet({'data' : images}, trainable=False)
        main = googLeNet.layers['main_branch']
        aux1 = googLeNet.layers['aux1_branch']
        aux2 = googLeNet.layers['aux2_branch']
        
        # initialization
        initialize1 = tf.global_variables_initializer()
        sess_GoogLeNet = tf.Session(graph=graph_GoogLeNet)
        #sess_GoogLeNet = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess_GoogLeNet.run(initialize1)
        # Load the data
        googLeNet.load('posenet.npy', sess_GoogLeNet)
        
        #saver_GoogLeNet = tf.train.Saver()
        
    #Build poseCNN
    with graph_PoseCNN.as_default():
        #place holder intermadiate output
        main_out = tf.placeholder(tf.float32, [batch_size, 7, 7, 1024], name = 'main')
        aux1_out = tf.placeholder(tf.float32, [batch_size, 4, 4, 128], name = 'aux1')
        aux2_out = tf.placeholder(tf.float32, [batch_size, 4, 4, 128], name = 'aux2')
        # Placeholder for output
        poses_x = tf.placeholder(tf.float32, [batch_size, 3], name = 'posex')
        poses_q = tf.placeholder(tf.float32, [batch_size, 4], name = 'poseq')
         # define network
        poseCNN = PoseCNN({'main_branch' : main_out, 'aux1_branch' : aux1_out, 'aux2_branch' : aux2_out})
        trans = poseCNN.layers['trans_out']
        rot = poseCNN.layers['rot_out']
        
        # Define Loss Function
        loss_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(trans, poses_x))))
        loss_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(rot, poses_q))))*10
        loss = loss_x + loss_q
        opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)

        
        # initialization
        initialize2 = tf.global_variables_initializer()
        sess_PoseCNN = tf.Session(graph=graph_PoseCNN)
        #sess_PoseCNN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess_PoseCNN.run(initialize2)
        
        saver = tf.train.Saver()
        
    #array to store intermadiate output
    main1 = np.zeros((batch_size,7,7,1024), dtype=np.float32)
    main2 = np.zeros((batch_size,7,7,1024), dtype=np.float32)
       
    aux1_1 = np.zeros((batch_size,4,4,128), dtype=np.float32)
    aux1_2 = np.zeros((batch_size,4,4,128), dtype=np.float32)
       
    aux2_1 = np.zeros((batch_size,4,4,128), dtype=np.float32)
    aux2_2 = np.zeros((batch_size,4,4,128), dtype=np.float32)
    
    loss_history = np.zeros((1,4))
    for i in range(max_iterations):
        # generating next batch
        image1, image2, np_trans, np_rot = gen_data_batch(datasource)
        feed1 = {images: image1}
        feed2 = {images: image2}
        
        with graph_GoogLeNet.as_default():
            aux1_2, aux1_1, main1 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed1)
            aux2_2, aux2_1, main2 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed2)
        
        # Element-wise Product
        aux1_1 = np.multiply(aux1_1, aux2_1)
        aux1_2 = np.multiply(aux1_2, aux2_2)
        main1 = np.multiply(main1, main2)
        # Prepare intermidaiate input for poseCNN
        feed_pose = {main_out : main1, aux1_out : aux1_1, aux2_out : aux1_2, poses_x : np_trans, poses_q : np_rot}
        
        with graph_PoseCNN.as_default():
            # Back-Prop
            sess_PoseCNN.run(opt, feed_dict=feed_pose)
            # Evaluate
            batch_loss, b_l_x, b_l_q = sess_PoseCNN.run([loss, loss_x, loss_q], feed_dict=feed_pose)
            loss_history = np.append(loss_history, [(i, batch_loss, b_l_x, b_l_q)], axis=0)
        if i % 20 == 0:
            print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(batch_loss))
            np.save('loss_over_iteration.npy', loss_history)
        if i % 2000 == 0:
            saver.save(sess_PoseCNN, outputFile)
            print("Intermediate file saved at: " + outputFile)
            
    saver.save(sess_PoseCNN, outputFile)
    print("Intermediate file saved at: " + outputFile)


if __name__ == '__main__':
	main()
