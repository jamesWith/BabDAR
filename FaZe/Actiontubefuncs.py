import numpy as np
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import time

#def getlabelbboxlists(path, filename):
#	bucketlist = []
#	baboonlist = []
#	
#	with open(path+ filename + '.txt', 'r') as label:
#			labelstr = label.readlines()
#			for line  in labelstr:
#				line = line.split(' ')
#				if line[0] == '1':
#					line.remove('1')
#					bucketlist.append(line)
#				else
#					line.remove('0')
#					baboonlist.append(line)
#	return(baboonlist, bucketlist)

def imShow(image):
	get_ipython().run_line_magic('matplotlib', 'inline')
	plt.axis("off")
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()



def getdetectionbboxlists(path):
	bucketlist = []
	baboonlist = []
	baboonidlist = []
	firstframelist = []
	
	with open(path, 'r') as label:
		labelstr = label.readlines()
		for line  in labelstr:
			line = line.split(',')
			while(len(baboonlist)< int(line[0])): # add empty gaps for empty frames
				baboonlist.append([])
				bucketlist.append([])
			if int(line[7]) == 0: # if a baboon
				if int(line[1]) not in baboonidlist:
					baboonidlist.append(int(line[1]))
					firstframelist.append(int(line[0]))
				baboonlist[-1].append([int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[1])])#[left, top, width, height, baboon ID] in pixels
			else:
				bucketlist[-1].append([int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[1])])
	return(firstframelist, baboonidlist, baboonlist, bucketlist)


def getdetectionbboxlists2(path):
	bucketlist = []
	baboonlist = []
	baboonidlist = []
	firstframelist = []
	
	with open(path, 'r') as label:
		labelstr = label.readlines()
		firstframe = int(labelstr[0].split(',')[0])
		for line  in labelstr:
			line = line.split(',')
			while(len(baboonlist)< int(line[0])): # add empty gaps for empty frames
				baboonlist.append([])
				bucketlist.append([])
			if int(line[7]) == 0: # if a baboon
				baboonlist[-1].append([int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[1])])#[left, top, width, height, baboon ID] in pixels
			else:
				bucketlist[-1].append([int(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[1])])
	return(firstframe, baboonlist, bucketlist)

def calcdistance(centre1, centre2):
	xdist = abs(centre1[0] - centre2[0])
	ydist = abs(centre1[1] - centre2[1])
	dist = (xdist**2 + ydist**2)**0.5
	return (dist)

def centre(box):
	centrefromleft = box[0] + box[2]/2
	centrefromtop = box[1] + box[3]/2
	return (int(centrefromleft), int(centrefromtop))

def Intersecting(bbox1, bbox2): #bbox must be array in form [topleft distance from left, topleft distance from top, total width, total height] all values in pixels
	cl1 = bbox1[0] + (bbox1[2]/2)
	ct1 = bbox1[1] + (bbox1[3]/2)
	cl2 = bbox2[0] + (bbox2[2]/2)
	ct2 = bbox2[1] + (bbox2[3]/2)
	if (abs(cl1-cl2) <= (1.2*((bbox1[2] + bbox2[2])/2))) and (abs(ct1-ct2) <= (1.2*((bbox1[3] + bbox2[3])/2))):
		return True
	else:
		return False

def intersecting_area(detection, box):
	detbottom = detection[1] + detection[3]
	detleft = detection[0]
	detright = detection[0] + detection[2]
	boxbottom = box[1] + box[3]
	boxcentre = box[0] + box[2]/2
	dist = min(calcdistance([detleft, detbottom], [boxcentre, boxbottom]),calcdistance([detright, detbottom], [boxcentre, boxbottom]))

	return dist


def convertbboxtopixeltopleft(bbox, imgwidth, imgheight): 	# input fraction of image[centre from left, from top, width, height]
	left = (bbox[0]-bbox[2]/2)*imgwidth
	width = bbox[2]*imgwidth
	top = (bbox[1]-bbox[3]/2)*imgheight
	height = bbox[3]*imgheight
	return [left, top, width, height]


def getbucketcrop(bucket, frame):
	bucleft = bucket[0]
	buctop = bucket[1]
	bucright = bucket[0] + bucket[2]
	bucbot = bucket[1] + bucket[3]
	bucketcrop = np.asanyarray(frame)[buctop:bucbot, bucleft:bucright]
	return bucketcrop


def getbucketnumbers(bucketlist, cap):
	bucketdict = {}
	for framenum, bucketperframe in enumerate(bucketlist):
		for bucket in bucketperframe:
			if bucket[4] not in bucketdict:
				cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
				ret, currentframe = cap.read()
				imShow(getbucketcrop(bucket, currentframe))
				print('\n\n\n\n\n\n\n\n\n\n')
				time.sleep(4)
				bucketnumber = input('Enter bucket number (if obscured press enter to move on: ')
				if bucketnumber != '':
					bucketcolour = input('Enter bucket colour: ')
					bucketdict[bucket[4]] = bucketnumber + ' ' + bucketcolour
	return bucketdict




def Createcrop(frame, baboon, intersectinglist): #bbox as [left, top, width, height, baboon ID] in pixels
	# crop frame to around baboon
	# create black canvas
	# create mask to only show good bits
	framelist =[]
	#frame = cv2.line(frame, bucketcentre, babooncentre, (255,255,255), 3)
	cropsize = 64*4
	#for baboon in baboonbox:
	endframe = np.full((1080, 1920, 3), 128 ,dtype=np.uint8)
	mask = np.zeros([1080, 1920, 3],dtype=np.uint8)
	left = baboon[0]
	right = baboon[0] + baboon[2]
	top = baboon[1]
	bottom = baboon[1] + baboon[3]


	imgsize = max((bottom - top), (right - left)) * 1.2
	bottom = min(int(top + imgsize),1080)
	centre = ((right+left)/2)
	left = max(int(centre-(imgsize/2)), 0) - (max(int(centre+(imgsize/2)), 1920) - 1920)
	right = min(int(centre+(imgsize/2)), 1920) - min(int(centre-(imgsize/2)), 0)
	scale = imgsize/cropsize

	mask = cv2.rectangle(mask, (int(baboon[0]-0.2*imgsize),baboon[1]), (int(baboon[0] + baboon[2]+0.2*imgsize),baboon[1] + baboon[3]), (255,255,255), -1)


	for bucket in intersectinglist:
		mask = cv2.rectangle(mask, (bucket[0],bucket[1]), (bucket[0] + bucket[2],bucket[1] + bucket[3]), (255,255,255), -1)

	mask = mask == 255
	endframe[mask] = frame[mask]
	crop_frame = endframe[top:bottom, left:right]
	crop_frame = cv2.resize(crop_frame, (cropsize, cropsize))
	#cv2.imshow('image',crop_frame)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return crop_frame

def Createcropstabilised(frame, baboon, intersectinglist, centre, imgsize): #bbox as [left, top, width, height, baboon ID] in pixels
	# crop frame to around baboon
	# create gray canvas
	# create mask to only show good bits
	framelist =[]
	#frame = cv2.line(frame, bucketcentre, babooncentre, (255,255,255), 3)
	cropsize = 64*4

	endframe = np.full((1080, 1920, 3), 128 ,dtype=np.uint8)
	mask = np.zeros([1080, 1920, 3],dtype=np.uint8)

	imgsize = imgsize * 1.5
	top = max(int(centre[1]-(imgsize/2)), 0) - (max(int(centre[1]+(imgsize/2)), 1080) - 1080)
	bottom = min(int(centre[1]+(imgsize/2)), 1080) - min(int(centre[1]-(imgsize/2)), 0)
	left = max(int(centre[0]-(imgsize/2)), 0) - (max(int(centre[0]+(imgsize/2)), 1920) - 1920)
	right = min(int(centre[0]+(imgsize/2)), 1920) - min(int(centre[0]-(imgsize/2)), 0)
	scale = imgsize/cropsize

	mask = cv2.rectangle(mask, (int(baboon[0]-0.2*baboon[2]),baboon[1]), (int(baboon[0] + baboon[2]+0.2*baboon[2]),baboon[1] + baboon[3]), (255,255,255), -1)


	for bucket in intersectinglist:
		mask = cv2.rectangle(mask, (bucket[0],bucket[1]), (bucket[0] + bucket[2],bucket[1] + bucket[3]), (255,255,255), -1)

	mask = mask == 255
	endframe[mask] = frame[mask]
	crop_frame = endframe[top:bottom, left:right]
	crop_frame = cv2.resize(crop_frame, (cropsize, cropsize))

	return crop_frame, top, left, scale

