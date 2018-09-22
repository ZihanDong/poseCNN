# Import the converted model's class
import sys
sys.path.append('../src')
import numpy as np
import argparse
import random
import tensorflow as tf
from GoogLeNet import GoogLeNet
from PoseCNN2 import PoseCNN
import cv2
from tqdm import tqdm

test_samples = 15000
batch_size = 300
result_folder = '../result/'
setting_list = ['new_CNN/', 'loss_CNN/', 'deep_CNN/', 'deep_CNN_iter300/', 'deep_CNN_batch20/', 'deep_CNN_iter30000/', 'deep_deeper/']
# Set this path to your dataset directory
data_idx = '_data_mvs.txt'

class datasource(object):
    def __init__(self, images1, images2, poses, idx, max_size, categ):
        self.images1 = images1
        self.images2 = images2
        self.poses = poses
        self.max_size = max_size
        self.idx = idx
        self.pos = 0
        self.categ = categ


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

def get_data(mode = 'test', sub_sample = 'down_sampled', illumination = 'max'):
    poses = []
    images1 = []
    images2 = []
    categ = []

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
            categ.append(int(categoryId))
            imgFiledId1 = '0'+imgFiledId1 if len(imgFiledId1)==1 else imgFiledId1
            imgFiledId2 = '0'+imgFiledId2 if len(imgFiledId2)==1 else imgFiledId2
            images1.append('D:/Dataset/' + sub_sample + '/scan'+categoryId+'/'\
                           +'clean_0' + imgFiledId1 + '_max.png')
            images2.append('D:/Dataset/' + sub_sample + '/scan'+categoryId+'/'\
                           +'clean_0' + imgFiledId2 + '_max.png')
    max_size = len(poses)
    indices = list(range(max_size))
    random.shuffle(indices)
    return datasource(images1, images2, poses, indices, max_size, categ)

def gen_data_batch(source):
    image1_batch = []
    image2_batch = []
    pose_x_batch = []
    pose_q_batch = []
    categ = []
    for i in range(batch_size):
        pos = i + source.pos
        pose_x = source.poses[source.idx[pos]][0:3]
        pose_q = source.poses[source.idx[pos]][3:7]
        image1_batch.append(source.images1[source.idx[pos]])
        image2_batch.append(source.images2[source.idx[pos]])
        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)
        categ.append(source.categ[source.idx[pos]])
    image1_batch = preprocess(image1_batch)
    image2_batch = preprocess(image2_batch)
    source.pos += i
    if source.pos + i > source.max_size:
        source.pos = 0
    return image1_batch, image2_batch, np.array(pose_x_batch), np.array(pose_q_batch), np.array(categ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('select', type=int)
    args = parser.parse_args()
    settings = setting_list[args.select]
    print('Evaluating ' + setting_list[args.select] + '  ......')
    # Create 2 separate graphs
    graph_GoogLeNet = tf.Graph()
    graph_PoseCNN = tf.Graph()
    #outputFile_GoogLeNet = "GoogLeNet.ckpt"
    test_data = get_data(mode = 'test', sub_sample = 'down_sampled')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #Build GoogLeNet
    with graph_GoogLeNet.as_default():
        #place holder for input
        images = tf.placeholder(tf.float32, [1, 224, 224, 3], name = 'input')
        
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
        main_out = tf.placeholder(tf.float32, [1, 7, 7, 1024], name = 'main')
        aux1_out = tf.placeholder(tf.float32, [1, 4, 4, 128], name = 'aux1')
        aux2_out = tf.placeholder(tf.float32, [1, 4, 4, 128], name = 'aux2')

         # define network
        poseCNN = PoseCNN({'main_branch' : main_out, 'aux1_branch' : aux1_out, 'aux2_branch' : aux2_out})
        trans = poseCNN.layers['trans_out']
        rot = poseCNN.layers['rot_out']
        
        
        # initialization
        initialize2 = tf.global_variables_initializer()
        sess_PoseCNN = tf.Session(graph=graph_PoseCNN)
        #sess_PoseCNN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.Saver()
        sess_PoseCNN.run(initialize2)
        #saver = tf.train.import_meta_graph('PoseCNN.ckpt.meta')
        saver.restore(sess_PoseCNN, result_folder + settings + 'PoseCNN.ckpt')
    
    #array to store intermadiate output
    main1 = np.zeros((1,7,7,1024), dtype=np.float32)
    main2 = np.zeros((1,7,7,1024), dtype=np.float32)
       
    aux1_1 = np.zeros((1,4,4,128), dtype=np.float32)
    aux1_2 = np.zeros((1,4,4,128), dtype=np.float32)
       
    aux2_1 = np.zeros((1,4,4,128), dtype=np.float32)
    aux2_2 = np.zeros((1,4,4,128), dtype=np.float32)
    
    
    # Here we use batch data to save memory
    predictions = np.zeros((test_samples, 15)) # Store prediction and ground truth
    print('Evaluating model with test data....')
    for n in tqdm(range(int(test_samples/batch_size))):
        image1, image2, np_trans, np_rot, np_categ = gen_data_batch(test_data)
        for i in range(batch_size):
            # generating next batch
            feed1 = {images: image1[i][np.newaxis,:]}
            feed2 = {images: image2[i][np.newaxis,:]}
            
            result = np.array([np_categ[i]])
            with graph_GoogLeNet.as_default():
                aux1_2, aux1_1, main1 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed1)
                aux2_2, aux2_1, main2 = sess_GoogLeNet.run([aux2, aux1, main], feed_dict=feed2)
           
            #np.savez('feature.npz', main=main, aux1=aux1_1, aux2=aux1_2)
            # Element-wise Product
            aux1_1 = np.multiply(aux1_1, aux2_1)
            aux1_2 = np.multiply(aux1_2, aux2_2)
            main1 = np.multiply(main1, main2)
            # Prepare intermidaiate input for poseCNN
            feed_pose = {main_out : main1, aux1_out : aux1_1, aux2_out : aux1_2}
            
            with graph_PoseCNN.as_default():
                pose_q= np.asarray(test_data.poses[i][3:7])
                pose_x= np.asarray(test_data.poses[i][0:3])
                predicted_x, predicted_q = sess_PoseCNN.run([trans, rot], feed_dict=feed_pose)
    
            # Store the prediction
            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)
                
            result = np.append(np.append(np.append(np.append(result, np_trans[i, :]), np_rot[i, :]), predicted_x), predicted_q)
            predictions[int(n*batch_size+i),:] = result
            
        np.save(result_folder + settings + 'prediction.npy', predictions)
    print(settings + 'Test finished, please load the predicted pose in Visualization and Evaluation tools for further evaluation')
    
if __name__ == '__main__':
	main()
