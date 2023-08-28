from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import werkzeug

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most logs
from io import BytesIO
from absl import flags
import tarfile
from six.moves import urllib
import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TF logs
import sys
import cv2
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from ModelFiles.src.util import image as img_util
from ModelFiles.src.util import openpose as op_util
from ModelFiles.src.RunModel import RunModel
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import sys
import os
import ModelFiles.src.utils as utils
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

DATA_DIR = "ModelFiles/data"
flags.DEFINE_string('img_path', 'data/k3.png', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
UPLOAD_FOLDER = 'ModelFiles\\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		#"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()

		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		  image: A PIL.Image object, raw input image.

		Returns:
		  resized_image: RGB image resized from original input image.
		  seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

def preprocess_image(img_path, json_path=None):
    img = img_path#io.imread(img_path)
    # print("\n$$$$$$$",img.shape)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != 224:
#            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(224) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               224)
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def calc_measure(cp, vertex):#, facet):
  measure_list = []
  
  for measure in cp:
    length = 0.0
    p2 = vertex[int(measure[0][1]), :]
    for i in range(0, len(measure)):#1
      p1 = p2
      if measure[i][0] == 1:
        p2 = vertex[int(measure[i][1]), :]  
      elif measure[i][0] == 2:
        p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
        vertex[int(measure[i][2]), :] * measure[i][4]
#        print("if 2 Measurement",int(measure[i][1]))   
      else:
        p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
          vertex[int(measure[i][2]), :] * measure[i][5] + \
          vertex[int(measure[i][3]), :] * measure[i][6]
      length += np.sqrt(np.sum((p1 - p2)**2.0))

    measure_list.append(length * 100)# * 1000

  measure_list[8] = measure_list[8] * 0.36#reducing the error in measurement added due to unarranged vertices
  measure_list[3] = measure_list[3] * 0.6927
  return np.array(measure_list).reshape(utils.M_NUM, 1)

def convert_cp():
    
  f = open(os.path.join(DATA_DIR, 'customBodyPoints.txt'), "r")

  tmplist = []
  cp = []
  for line in f:
    if '#' in line:
      if len(tmplist) != 0:
        cp.append(tmplist)
        tmplist = []
    elif len(line.split()) == 1:
      continue
    else:
      tmplist.append(list(map(float, line.strip().split())))
  cp.append(tmplist)


  return cp


    


def process_file(filename):
    image = Image.open(filename)

    
    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]
    model_dir="./ModelFiles/deeplab_model/"

    if not os.path.exists(model_dir):
        tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], 
			     download_path)
        print('download completed! loading DeepLab model...')

    model_path="./ModelFiles/deeplab_model/deeplabv3_pascal_trainval_2018_01_04.tar.gz"

    MODEL = DeepLabModel(model_path)
    print('model loaded successfully!')
        
    res_im, seg = MODEL.run(image)
    seg = cv2.resize(seg.astype(np.uint8), image.size)
    mask_sel = (seg == 15).astype(np.float32)
    mask = 255 * mask_sel.astype(np.uint8)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    res = cv2.bitwise_and(img, img, mask=mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    measurements = get_measurements(bg_removed, None)
    
    return measurements


def get_measurements(bg_removed, json_path=None):
    sess = tf.Session()
    model = RunModel(sess=sess)

    input_img, proc_param, img = preprocess_image(bg_removed, json_path)
    input_img = np.expand_dims(input_img, 0)
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    return extract_measurements(verts[0])


def extract_measurements(vertices):
    genders = ["male"]
    measure_dict = {}
    for gender in genders:
        cp = convert_cp()
        measure = calc_measure(cp, vertices)
        print("Prediction Results: \n")
        for i in range(0, utils.M_NUM):
            formatted_value = "{:.2f}".format(measure[i].item())
            formatted_value_float = float(formatted_value)
            print("%s: %.2f cm" % (utils.M_STR[i], formatted_value_float))
            measure_dict[utils.M_STR[i]] = f"{formatted_value_float} cm"

    face_path = './ModelFiles/src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = 'test.obj'
    with open(obj_mesh_name, 'w') as fp:
        for v in vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
    print("3D Model Has been Saved in test.obj... use https://3dviewer.net/ to view the 3d model of the image.")
    return measure_dict

@app.route('/', methods=['GET'])
def home():
    return "This is the root directory of the Prediction Model\n Use /predict/ with the link to go to the model"

@app.route('/predict/', methods=['GET'])
def predict():
    return "This is the model page.\n Use post request with an image so that model can do prediction"

@app.route('/predict/', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print("No image part in the request.")
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    print("File object recieved.")
    if file.filename == '':
        print("No file selected.")
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(UPLOAD_FOLDER, 'upload1.jpg')
        file.save(filename)
        # Now that the file is saved, process it
        result = process_file(filename)

        # Delete the uploaded file
        if os.path.exists(filename):
            os.remove(filename)
            print("File removed:", filename)

        return jsonify(result)
    
    else:
        return jsonify({'error': 'File upload failed'}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
