# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json

import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
from shapely.geometry import Polygon

class labelme2coco(object):
	def __init__(self,labelme_json=[],save_json_path='./new.json'):
		'''
		:param labelme_json: all the labelme json files
		:param save_json_path: the dir of saved json files
		'''
		self.labelme_json=labelme_json
		self.save_json_path=save_json_path
		self.images=[]
		self.categories=[]
		self.annotations=[]
		# self.data_coco = {}
		self.label=[]
		self.annID=1
		self.height=0
		self.width=0
		
		self.save_json()
		
		
	def data_transfer(self):
		for num,json_file in enumerate(self.labelme_json):
			print(json_file,num)
			with open(json_file,'r') as fp:
				data = json.load(fp)  # load the json files
				#print(data)
				self.images.append(self.image(data,num))
				keypoints = []
				val = 0
				full_label = ''
				points = None
				for shapes in data['shapes']:
					full_label=shapes['label']		
					#print(full_label)
					if(full_label == "taptopleft"):
						val = 1
					if(full_label == "taptopmid"):
						val = 2
					if(full_label == "taptopright"):
						val = 3
					if full_label == "tapbottomleft":
						val = 4
					if full_label == "tapbottomright":
						val = 5
					if full_label == "squidleft":
						val = 6
					if full_label == "squidright":
						val = 7
					if full_label == "bottle":
						val = 8
					#print("values ", val)
					
					if full_label not in self.label:
						self.categories.append(self.categorie(full_label))
						self.label.append(full_label)
					
					shapetype = shapes['shape_type']
					#print(shapetype)
					
					if shapetype == "point":
						keypoints.append(shapes['points'][0][0])
						keypoints.append(shapes['points'][0][1])
						keypoints.append(val)
						#print(keypoints)
					
					if shapetype == "rectangle":
						points=shapes['points']
						#print(points)
					
				self.annotations.append(self.annotation(points,full_label,num,keypoints))
				self.annID+=1
        
		print (num, json_file)

	def image(self,data,num):
		image={}
		#print("imagedata: ",data['imageData'])
		#img = utils.img_b64_to_arr(data['imageData']) 

		height = data['imageHeight']
		width = data['imageWidth']
		img = None
		image['height']=height
		image['width'] = width
		image['id']=num+1
		image['file_name'] = data['imagePath'].split('/')[-1]

		self.height=height
		self.width=width

		return image


	def categorie(self,label):
		categorie={}
		categorie['supercategory'] = "bottle"
		categorie['id']=len(self.label)+1 # 0 is default as background
		categorie['name'] = label
		return categorie


	def annotation(self,points,label,num,keypoints):
		annotation={}
		annotation['segmentation']=''#[np.asarray(keypoints).flatten().tolist()]
		annotation['keypoints']=keypoints
		annotation['num_keypoints']=int(len(keypoints)/3)
		annotation['iscrowd'] = 0
		annotation['image_id'] = num+1

		annotation['bbox'] = list(map(float,self.getbbox(points)))

		annotation['category_id'] = self.getcatid(label)
		annotation['id'] = self.annID

		# Get the area value
		area = round((points[1][0]-points[0][0])*(points[1][1]-points[0][1]), 6)
		annotation['area']=area
		return annotation


	def getcatid(self,label):
		for categorie in self.categories:
			if label==categorie['name']:
				return categorie['id']
		return -1


	def getbbox(self,points):

		polygons = points
		mask = self.polygons_to_mask([self.height,self.width], polygons)
		return self.mask2box(mask)

	def mask2box(self, mask):
		'''从mask反算出其边框
		mask：[h,w]  0、1组成的图片
		1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
		'''
		# np.where(mask==1)
		index = np.argwhere(mask == 1)
		rows = index[:, 0]
		clos = index[:, 1]
		# 解析左上角行列号
		left_top_r = np.min(rows)  # y
		left_top_c = np.min(clos)  # x

		# 解析右下角行列号
		right_bottom_r = np.max(rows)
		right_bottom_c = np.max(clos)

		# return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
		# return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
		# return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
		return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式


	def polygons_to_mask(self,img_shape, polygons):
		mask = np.zeros(img_shape, dtype=np.uint8)
		mask = PIL.Image.fromarray(mask)
		xy = list(map(tuple, polygons))
		PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
		mask = np.array(mask, dtype=bool)
		return mask


	def data2coco(self):
		data_coco={}
		data_coco['images']=self.images
		data_coco['categories']=self.categories
		data_coco['annotations']=self.annotations
		return data_coco


	def save_json(self):
		self.data_transfer()
		self.data_coco = self.data2coco()

		json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4

# convert train data set json files
labelme_json=glob.glob(r'G:\squidtap\keypoint1\*.json')
labelme2coco(labelme_json,'squidtap_train.json')

# Convert test data set json files
#labelme_json=glob.glob(r'D:\code\agriculture\wheat\dataset\mask rcnn model\2 Labeled data\val_data\json\*.json')
#labelme2coco(labelme_json,'./wheat_spike_val.json')
