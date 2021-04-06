import os 
import time

import torch
import torch.nn.parallel
import torchvision
import cv2
import torchvision.transforms as Transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchcam.cams import CAM

from Actiontubefuncs import *
from Modified_CNN import TSN_model
from transforms import *
from var_evaluation import Evaluation
from misc_functions import get_example_params, save_class_activation_images

import argparse

parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'baboons'])
parser.add_argument('filename',type=str)

parser.add_argument('weights', nargs='+', type=str,
                    help='1st and 2nd index is RGB and RGBDiff weights respectively')
parser.add_argument('--det_file_name', type=str, default='dettrack.txt')
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=8, help="Sliding Window Width")
parser.add_argument('--sampling_freq', type=int, default=12, help="Take 1 image every 12 image")
parser.add_argument('--delta', type=int, default=2, help="Sliding Window Delta")
parser.add_argument('--psi', type=float, default=2.5)
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--classInd_file', type=str, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--score_weights', nargs='+', type=float, default=[1,1.5])
parser.add_argument('--quality',type=str,default = '480p')
parser.add_argument('--colab', type=bool, default=False)


args = parser.parse_args()

#this function returns a dictionary (keys are string label numbers & values are action labels)
def label_dic(classInd):
  action_label={}
  with open(classInd) as f:
      content = f.readlines()
      content = [x.strip('\r\n') for x in content]
  f.close()

  for line in content:
      label, action = line.split(' ')
      if action not in action_label.keys():
          action_label[label] = action
          
  return action_label

  #The following 2 function responsible for display a white rectangler around any text displayed on OpenCV winodw
def add_status(frame_,s=(),x=5,y=12,font = cv2.FONT_HERSHEY_SIMPLEX
    ,fontScale = 0.4,fontcolor=(255,255,255),thickness=3,box_flag=True
    ,alpha = 0.4,boxcolor=(129, 129, 129),x_mode=None):
    
    if x_mode is 'center':
        x=frame_.shape[1]//2
    elif x_mode is 'left':
        x=frame_.shape[1]

    origin=np.array([x,y])
    y_c = add_box(frame=frame_ ,text=s ,origin=origin, font=font, fontScale=fontScale
        ,thickness=thickness ,alpha=alpha ,enable=box_flag,color=boxcolor,x_mode=x_mode)
    cv2.putText(frame_,s, tuple(origin) 
        ,font, fontScale, fontcolor, thickness)


def add_box(frame,text,origin,font,color,fontScale=1,thickness=1,alpha=0.4,enable=True,x_mode=None):
    box_dim = cv2.getTextSize(text,font,fontScale,thickness)
    if x_mode is 'center':
        origin[:] = origin - np.array([box_dim[0][0]//2,0])
    elif x_mode is 'left':
        origin[:] = origin - np.array([box_dim[0][0]+2,0])
    pt1 = origin - np.array([0,box_dim[0][1]])
    pt2 = pt1+box_dim[0]+np.array([0,box_dim[0][1]//4+thickness])
    if enable:
        overlay = frame.copy()
        cv2.rectangle(overlay,tuple(pt1),tuple(pt2),color,-1)  # A filled rectangle

        # Following line overlays transparent rectangle over the image
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return pt2[1]-pt1[1]+1
  
#intialize tensors to save some of the scores for the next iteration
#now part of cropobj object
#pre_scoresRGB  = torch.zeros((args.num_segments - args.delta ,2)).cuda()
#pre_scoresRGBDiff =  torch.zeros((args.num_segments - args.delta ,2)).cuda()

def eval_video(crop, data, model):
      """
      Evaluate single video
      video_data : (data in shape (1,num_segments*length,H,W))
      return     : a tensor of (2) size representing a score for certain batch of frames
      """
      with torch.no_grad():
          #reshape data to be in shape of (num_segments,length,H,W)
          #Forword Propagation
          if model == 'RGB':
              input = data.view(-1, 3, data.size(1), data.size(2))
              #concatenate the new scores with previous ones (Sliding window)
              output = torch.cat((crop.pre_scoresRGB,model_RGB(input)),0)
              #Save Current scores as previous ones for the next iteration
              crop.pre_scoresRGB = output.data[-(args.num_segments - args.delta):,]
              
          elif model == 'RGBDiff':
              input = data.view(-1, 18, data.size(1), data.size(2))
              output =torch.cat((crop.pre_scoresRGBDiff,model_RGBDiff(input)),0)
              crop.pre_scoresRGBDiff = output.data[-(args.num_segments - args.delta):,]
              
          output_tensor = output.data.mean(dim = 0,keepdim=True)
          
      return output_tensor
  
if args.dataset == 'ucf101':
  num_class = 101
elif args.dataset == 'baboons':
  num_class = 2
else:
  raise ValueError('Unkown dataset: ' + args.dataset)

#Intializing the streams  
model_RGB = TSN_model(num_class, 1, 'RGB',
                base_model_name=args.arch, consensus_type='avg', dropout=args.dropout)

model_RGBDiff = TSN_model(num_class, 1, 'RGBDiff',
                base_model_name=args.arch, consensus_type='avg', dropout=args.dropout)
  
for i in range(len(args.weights)):
  #load the weights of your model training
  if args.gpus is not None:
  	checkpoint = torch.load(args.weights[i])
  else:
  	checkpoint = torch.load(args.weights[i], map_location='cpu')
  print("epoch {}, best acc1@: {}" .format(checkpoint['epoch'], checkpoint['best_acc1']))

  base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
  if i==0:
      model_RGB.load_state_dict(base_dict)
  else:
      model_RGBDiff.load_state_dict(base_dict)

  #Required transformations
transform = torchvision.transforms.Compose([
       GroupScale(model_RGB.scale_size),
       GroupCenterCrop(model_RGB.input_size),
       Stack(roll=args.arch == 'BNInception'),
       ToTorchFormatTensor(div=args.arch != 'BNInception'),
       GroupNormalize(model_RGB.input_mean, model_RGB.input_std),
               ])


if args.gpus is not None:
  devices = [args.gpus[i] for i in range(args.workers)]
  model_RGB = torch.nn.DataParallel(model_RGB.cuda(devices[0]), device_ids=devices)
  model_RGBDiff = torch.nn.DataParallel(model_RGBDiff.cuda(devices[0]), device_ids=devices)
else:
  devices = list(range(args.workers))


model_RGB.eval()
model_RGBDiff.eval()    


class Cropobj(object):
	def __init__(self, BaboonDetection, nearestbucket, size = 0, centreleft = 0, centretop = 0 ):
		fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
		
		self.missedframes = 0 # how many frames its missed
		self.notintersecting = 0
		self.ID = BaboonDetection[4]
		self.lastpos = 0
		self.length = -1
		if size == 0:
			self.size = max(BaboonDetection[2] , BaboonDetection[3])
			self.centreleft, self.centretop = centre(BaboonDetection)
		else:
			self.size = size
			self.centreleft = centreleft
			self.centretop = centretop
		self.prevnearestbucket = nearestbucket
		self.overlapframes = []
		self.framesforrecognition = []
		self.intersectingdetails = []
		if args.gpus is None:
			self.pre_scoresRGB  = torch.zeros((args.num_segments - args.delta ,2))
			self.pre_scoresRGBDiff =  torch.zeros((args.num_segments - args.delta ,2))
		else:
			self.pre_scoresRGB  = torch.zeros((args.num_segments - args.delta ,2)).cuda()
			self.pre_scoresRGBDiff =  torch.zeros((args.num_segments - args.delta ,2)).cuda()
		cropcount = 0


def Add_new_crops(crops, detections, bucketlist, framenum):
	for baboon in detections: #Go through all detections

		if not any(crop for crop in crops if crop.ID == baboon[4]): # If a crop doesnt already exist for this ID

			cropneeded = False
			for bucket in bucketlist[framenum]: # Loop through buckets
				if Intersecting(baboon, bucket): # Find if the baboon is intesecting any buckets
					cropneeded = True

			if cropneeded:
				intersectinglist = []
				minbucketdist = 1920
				nearestbucket = -1

				for bucket in bucketlist[framenum]:
					dist = calcdistance(centre(bucket) , centre(baboon))

					if dist < minbucketdist: # Save the closest bucket
						minbucketdist = dist
						nearestbucket = bucket[4]

				newcrop = Cropobj(baboon, nearestbucket)
				crops.append(newcrop) # Add to list of crops
	return

def Create_action_tubes(crop, crops, detections, bucketlist, framenum, currentframe, maxmissedframes, maxnotintersecting):

	crop.length += 1
	intersectinglist = []
	bucketdetails = []

	detection = [detection for detection in detections if detection[4] == crop.ID] # Returns the the detection of the current crop
	if detection != []:
		detection = detection[0]
		
		for bucket in bucketlist[framenum]:
			if Intersecting(detection, bucket): # Find if the baboon is intesecting any buckets
				intersectinglist.append(bucket)
				bucketdetails.append([bucket, distance_score(detection, bucket)])

		crop.centreleft += (centre(detection)[0] - crop.centreleft)*0.1
		crop.centretop += (centre(detection)[1] - crop.centretop)*0.1
		crop.size += (max(detection[2] , detection[3]) - crop.size)*0.05
		frame, top, left, scale = Createcropstabilised(currentframe, detection, intersectinglist, [crop.centreleft,crop.centretop] , crop.size)
		
		crop.lastpos = detection

		if intersectinglist != []:
			crop.notintersecting = 0
		else:
			crop.notintersecting += 1
			if crop.notintersecting >= maxnotintersecting:
				crops.remove(crop)
	else:
		frame, top, left, scale = Createcropstabilised(currentframe, crop.lastpos, bucketlist[framenum], [crop.centreleft,crop.centretop] , crop.size)
		
		crop.missedframes += 1
		if crop.missedframes >= maxmissedframes:
			crops.remove(crop)

	if crop.length % args.sampling_freq < 6:
		frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		crop.framesforrecognition.append(frame)
		crop.intersectingdetails.append(bucketdetails)
		#print('Picked frame, ', framenum)
		#crop.vidout.write(Createcrop(currentframe, crop.lastpos, bucketlist[framenum]))
		
	return

def Run_detection(crop, action_label):
	detected_action = ''
	frames = crop.framesforrecognition
	if len(frames) == args.delta*6:	
		if args.gpus is None:
			frames = transform(frames)
		else:
			frames = transform(frames).cuda()
		scores_RGB = eval_video(crop, frames[0:len(frames):6], 'RGB')[0,] 
		scores_RGBDiff = eval_video(crop, frames[:], 'RGBDiff')[0,]           
		#Fusion
		final_scores = args.score_weights[0]*scores_RGB + args.score_weights[1] * scores_RGBDiff
		#Just like np.argsort()[::-1]
		scores_indcies = torch.flip(torch.sort(final_scores.data)[1],[0])
		#Prepare List of top scores for action checker
		TopScoresList = []
		for i in scores_indcies[:2]:
			TopScoresList.append(int(final_scores[int(i)]))
		#Check for "no action state"
		action_checker = Evaluation(TopScoresList, args.psi)
		#action_checker = True
		if not action_checker:
			print('No Assigned Action')
		else:
			detected_action = action_label[str(int(scores_indcies[0]))]
		#for i in scores_indcies[:2]:
		#	print('%-22s %0.2f'% (action_label[str(int(i))], final_scores[int(i)]))
		#print('<----------------->')
		frames = []
	return detected_action


def Detect(filename):

	detfile = args.det_file_name
	
	# Video to generate crops
	cap = cv2.VideoCapture(filename)
	
	# Get: the first frame each baboon appears in, and for each Frame: the baboon ID in the frame, their location, the bucket location in the frame. 
	startframe, baboonlist, bucketlist = getdetectionbboxlists2(detfile)
	
	# Set the number of missed frames before a crop video stream is dropped
	maxmissedframes = 15
	maxnotintersecting = 60
	
	# move to the correct point in the video stream
	cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)

	crops = []
	preframes = []

	action_label = label_dic(args.classInd_file)

	with open(detfile[:-9] + "action.txt", 'w') as out_file:
			# Start looking for the baboon at first frame it appears
		for framenum, detections in enumerate(baboonlist[startframe::], start = startframe):
			ret, currentframe = cap.read()
			preframes.append(currentframe)
			if len(preframes) > 6:
				preframes.pop(0)
			
			if detections != []: # if baboons are detected
	
				detections = np.asarray(detections)
	
				Add_new_crops(crops, detections, bucketlist, framenum)
				
				for crop in crops:
					
					Create_action_tubes(crop, crops, detections, bucketlist, framenum, currentframe, maxmissedframes, maxnotintersecting)
					action = Run_detection(crop, action_label)
					
					if action == 'Taking_from_bucket':									
						bucketdict = {}

						for frameinsegment, bucketdetails in enumerate(crop.intersectingdetails[:-5]): # for each frame in the 6 frame segment except last frame
							for bucket in bucketdetails: # for each bucket in each frame
								buc1 = getbucketcrop(bucket[0], preframes[frameinsegment])
								buc2 = getbucketcrop(bucket[0], preframes[frameinsegment + 5])
								rgbdiffbucket = np.abs(np.subtract(buc1.astype(np.int16), buc2.astype(np.int16)))
								movevalue = np.mean(rgbdiffbucket)

								distvalue = bucket[1]
								if distvalue==0:
									distvalue = 1
								score = movevalue/distvalue
								if bucket[0][4] not in bucketdict:
									bucketdict[bucket[0][4]] = score
								else:
									bucketdict[bucket[0][4]] = bucketdict[bucket[0][4]] + score #add up the total intersecting_area() over six frames
						BucketID = -1
						maxscore = 0
						for key, value in bucketdict.items():
							if value > maxscore:
								BucketID = key # Bucket with the highest total is assigned the action
								maxscore = value
						print('Baboon ID:' + str(crop.ID))
						print('Took from bucket:' + str(BucketID))
						print('At frame number:' + str(startframe + framenum))
						print('At time:' + str((startframe + framenum)/25) + 's')
						print('<----------------->')
						print(str(crop.ID) + ' ' + str(BucketID) + ' ' + str((startframe + framenum)/25), file=out_file)
	
					if len(crop.framesforrecognition) == args.delta*6:
						crop.framesforrecognition = []
						crop.intersectingdetails = []

	bucketdict = getbucketnumbers(bucketlist, cap)
	#print(bucketdict)
	action_dets = np.loadtxt(detfile[:-9] + "action.txt" , delimiter=' ', dtype=str)
	bucketcolourdict = {}
	baboonvisitnumber = {}
	baboonprevbucket = {}
	with open(detfile[:-9] + "action.txt", 'w') as out_file:
		print('PAIR, INDIV, PREF, TIME, VISIT N', file=out_file)
		for action in action_dets:
			if (action[1] != '-1') and ((action[0] not in baboonprevbucket) or (baboonprevbucket[action[0]]!=bucketdict[int(action[1])])):
				if action[0] not in baboonvisitnumber:
					baboonvisitnumber[action[0]] = 1
				else:
					baboonvisitnumber[action[0]] = baboonvisitnumber[action[0]] + 1

				if bucketdict[int(action[1])].split()[1] not in bucketcolourdict:
					bucketcolourdict[bucketdict[int(action[1])].split()[1]] = input('What is the contents of ' + bucketdict[int(action[1])].split()[1] + ' buckets?')

				print(bucketdict[int(action[1])].split()[0] + ',' + action[0]+ ','+ bucketcolourdict[bucketdict[int(action[1])].split()[1]] + ',' + action[2] + ',' + str(baboonvisitnumber[action[0]]), file=out_file)
			baboonprevbucket[action[0]] = bucketdict[int(action[1])]

	cap.release()
	return
			
if __name__ == '__main__':
	Detect(args.filename)
	
