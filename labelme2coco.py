# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import random
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
import shutil
from shapely.geometry import Polygon
import os

class labelme2coco(object):
	def __init__(self,labelme_json=[],train_json_path='./tain2017.json',val_json_path='./val2017.json',test_json_path='./test2017.json'):
		'''
		:param labelme_json: all the labelme json files
		:param train_json_path: the path of saved train json file
		:param val_json_path: the path of saved val json file 
		:param test_json_path: the path of saved test json file
		'''
		self.labelme_json=labelme_json
		self.train_json_path=train_json_path
		self.val_json_path=val_json_path
		self.test_json_path=test_json_path
		
		self.train_images=[]
		self.val_images=[]
		self.test_images=[]
		
		self.categories=[]
		
		self.train_annotations=[]
		self.val_annotations=[]
		self.test_annotations=[]
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
				oldfile = "G:\\squidtap\\json\\"+data['imagePath'].split('/')[-1]
				rand = random.randint(0, 9) #train [0,1,2,3,4,5],valid [6,7],test [8,9]
				newfile = ''
				if(rand>=0 and rand<=5):
					self.train_images.append(self.image(data,num))
					newfile = "G:\\squidtap\\images\\train2017\\"+data['imagePath'].split('/')[-1]
				elif(rand>=6 and rand<=7):
					self.val_images.append(self.image(data,num))
					newfile = "G:\\squidtap\\images\\val2017\\"+data['imagePath'].split('/')[-1]
				else:
					self.test_images.append(self.image(data,num))
					newfile = "G:\\squidtap\\images\\test2017\\"+data['imagePath'].split('/')[-1]
				shutil.move(oldfile, newfile)
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
					
					if ((full_label not in self.label) and full_label != "bottle"):
						self.categories.append(full_label)
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
				
				if(rand>=0 and rand<=5):
					self.train_annotations.append(self.annotation(points,full_label,num,keypoints))
				elif(rand>=6 and rand<=7):
					self.val_annotations.append(self.annotation(points,full_label,num,keypoints))
				else:
					self.test_annotations.append(self.annotation(points,full_label,num,keypoints))
				
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


	def annotation(self,points,label,num,keypoints):
		annotation={}
		annotation['segmentation']=''#[np.asarray(keypoints).flatten().tolist()]
		annotation['keypoints']=keypoints
		annotation['num_keypoints']=int(len(keypoints)/3)
		annotation['iscrowd'] = 0
		annotation['image_id'] = num+1

		annotation['bbox'] = list(map(int,self.getbbox(points)))

		annotation['category_id'] = 1
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
	
	
	def train_data2coco(self):
		data_coco={}
		data_coco['images']=self.train_images
		
		categorie={}
		categorie['supercategory'] = "bottle"
		categorie['id']=1 # 0 is default as background
		categorie['name'] = "bottle"
		categorie['keypoints'] = self.categories
		
		data_coco['categories']=[categorie]
		data_coco['annotations']=self.train_annotations
		return data_coco
		
	
	def val_data2coco(self):
		data_coco={}
		data_coco['images']=self.val_images
		
		categorie={}
		categorie['supercategory'] = "bottle"
		categorie['id']=1 # 0 is default as background
		categorie['name'] = "bottle"
		categorie['keypoints'] = self.categories
		
		data_coco['categories']=[categorie]
		data_coco['annotations']=self.val_annotations
		return data_coco


	def test_data2coco(self):
		data_coco={}
		data_coco['images']=self.test_images
		
		categorie={}
		categorie['supercategory'] = "bottle"
		categorie['id']=1 # 0 is default as background
		categorie['name'] = "bottle"
		categorie['keypoints'] = self.categories
		
		data_coco['categories']=[categorie]
		data_coco['annotations']=self.test_annotations
		return data_coco	


	def save_json(self):
		self.data_transfer()
		self.train_data_coco = self.train_data2coco()
		json.dump(self.train_data_coco, open(self.train_json_path, 'w'), indent=4)  # indent=4
		
		self.val_data_coco = self.val_data2coco()
		json.dump(self.val_data_coco, open(self.val_json_path, 'w'), indent=4)  # indent=4
		
		self.test_data_coco = self.test_data2coco()
		json.dump(self.test_data_coco, open(self.test_json_path, 'w'), indent=4)  # indent=4

# convert train data set json files
labelme_json=glob.glob(r'G:\squidtap\json\*.json')
#create image dir
if not os.path.exists(r'G:\\squidtap\images'):
	os.makedirs('G:\\squidtap\\images')

if not os.path.exists(r'G:\squidtap\images\train2017'):
	os.makedirs('G:\\squidtap\\images\\train2017')

if not os.path.exists(r'G:\squidtap\images\val2017'):
	os.makedirs('G:\\squidtap\\images\\val2017')

if not os.path.exists(r'G:\squidtap\images\test2017'):
	os.makedirs('G:\\squidtap\\images\\test2017')

#create annotation dir
if not os.path.exists(r'G:\squidtap\annotations'):
	os.makedirs('G:\\squidtap\\annotations')
	
labelme2coco(labelme_json,'annotations\squidtap_train2017.json','annotations\squidtap_val2017.json','annotations\image_info_test2017.json')

# Convert test data set json files
#labelme_json=glob.glob(r'D:\code\agriculture\wheat\dataset\mask rcnn model\2 Labeled data\val_data\json\*.json')
#labelme2coco(labelme_json,'./wheat_spike_val.json')
