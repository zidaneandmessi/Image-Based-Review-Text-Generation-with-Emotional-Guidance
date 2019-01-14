import pandas as pd
import numpy as np
import os
import pickle
from cnn_util import *

vgg_model = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

#annotation_path = './data/results_20130124.token'
flickr_image_path = './test'
feat_path = './test/test_feats.npy'
#annotation_result_path = './data/annotations.pickle'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=64, height=64)

#annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
#annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
#annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

#unique_images = annotations['image'].unique()
unique_images = flickr_image_path.join('test.jpg')
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

#annotations = pd.merge(annotations, image_df)
#annotations.to_pickle(annotation_result_path)

if not os.path.exists(feat_path):
    feats = cnn.get_features(unique_images, layers='conv5_3', layer_sizes=[256,4,4])
    np.save(feat_path, feats)

