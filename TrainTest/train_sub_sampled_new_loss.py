# Import the converted model's class
import sys
sys.path.append('../src')
import numpy as np
import random
import tensorflow as tf
from GoogLeNet import GoogLeNet
from PoseCNN import PoseCNN
import cv2

beta = 10
batch_size = 100
max_iterations = 5000
# Set this path to your dataset directory
data_idx = '_data_mvs.txt'
# Set this path to your working space directory
result_folder = '../result/'
settings = 'new_loss/'

class datasource(object):
    def __init__(self, images1, images2, poses, idx, max_size):
        self.images1 = images1
        self.images2 = images2
        self.poses = poses
        self.max_size = max_size
        self.idx = idx
        self.pos = 0

def preprocess(images):
    images_out = np.zeros((len(images), 3, 224, 224)) #final result
    for i in range(len(images)):
        temp_image = cv2.imread(images[i])
        images_out[i] = np.transpose(temp_image,(2,0,1))
    #compute images mean
    mean = np.mean(images_out, axis=0)

    #Subtract mean from all images
    images_out = np.transpose(images_out - mean, (0,2,3,1))
    return images_out

def get_data(mode = 'train', sub_sample = 'down_sampled', illumination = 'max'):
    poses = []
    images1 = []
    images2 = []

    with open('../data/'+ mode + data_idx) as f:
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
        pose_x = source.poses[source.idx[pos]][0:3]
        pose_q = source.poses[source.idx[pos]][3:7]
        image1_batch.append(source.images1[source.idx[pos]])
        image2_batch.append(source.images2[source.idx[pos]])
        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)
    image1_batch = preprocess(image1_batch)
    image2_batch = preprocess(image2_batch)
    source.pos += i
    if source.pos + i > source.max_size:
        source.pos = 0
    return image1_batch, image2_batch, np.array(pose_x_batch), np.array(pose_q_batch)


def main():
    print('Beta selected as :' + str(beta))
    # Create 2 separate graphs
    graph_GoogLeNet = tf.Graph()
    graph_PoseCNN = tf.Graph()
    #outputFile_GoogLeNet = "GoogLeNet.ckpt"
    train_data = get_data(mode = 'train')
    test_data = get_data(mode = 'test')
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
        googLeNet.load('../posenet.npy', sess_GoogLeNet)
        
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
        norm_rot = tf.transpose(tf.transpose(rot)/tf.norm(rot, axis=1))
        loss_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(norm_rot, poses_q))))*beta
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
    
    train_loss = np.zeros((1,4))
    test_loss = np.zeros((1,4))
    image1_t, image2_t, np_trans_t, np_rot_t = gen_data_batch(test_data)
    
    for i in range(max_iterations):
        # generating next batch
        image1, image2, np_trans, np_rot = gen_data_batch(train_data)
        feed1 = {images: image1}
        feed2 = {images: image2}
        
        aux1_2, aux1_1, main1 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed1)
        aux2_2, aux2_1, main2 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed2)
        
        # Element-wise Product
        aux1_1 = np.multiply(aux1_1, aux2_1)
        aux1_2 = np.multiply(aux1_2, aux2_2)
        main1 = np.multiply(main1, main2)
        # Prepare intermidaiate input for poseCNN
        feed_pose = {main_out : main1, aux1_out : aux1_1, aux2_out : aux1_2, poses_x : np_trans, poses_q : np_rot}
        
         # Back-Prop
        sess_PoseCNN.run(opt, feed_dict=feed_pose)
        if i % 10 == 0:
            # Evaluate
            batch_loss, b_l_x, b_l_q = sess_PoseCNN.run([loss, loss_x, loss_q], feed_dict=feed_pose)
            train_loss = np.append(train_loss, [(i, batch_loss, b_l_x, b_l_q)], axis=0)
            # Validation
            feed1 = {images: image1_t}
            feed2 = {images: image2_t}
            aux1_2, aux1_1, main1 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed1)
            aux2_2, aux2_1, main2 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed2)
            aux1_1 = np.multiply(aux1_1, aux2_1)
            aux1_2 = np.multiply(aux1_2, aux2_2)
            main1 = np.multiply(main1, main2)
            feed_pose = {main_out : main1, aux1_out : aux1_1, aux2_out : aux1_2, poses_x : np_trans, poses_q : np_rot}
            batch_loss_t, b_l_x, b_l_q = sess_PoseCNN.run([loss, loss_x, loss_q], feed_dict=feed_pose)
            test_loss = np.append(test_loss, [(i, batch_loss_t, b_l_x, b_l_q)], axis=0)
            
            print("iteration: " + str(i) + "\n\t" + "TrainLoss is: " + str(batch_loss) + "  TestLoss is:" + str(batch_loss_t))
            np.savez(result_folder + settings +'loss.npz', train = train_loss, test = test_loss)
        if i % 1000 == 0:
            saver.save(sess_PoseCNN, result_folder + settings + 'PoseCNN.ckpt')
            print("Intermediate file saved at: " + result_folder + settings + 'PoseCNN.ckpt')
            
    saver.save(sess_PoseCNN, result_folder + settings + 'PoseCNN.ckpt')
    print("Intermediate file saved at: " + result_folder + settings + 'PoseCNN.ckpt')



if __name__ == '__main__':
	main()
