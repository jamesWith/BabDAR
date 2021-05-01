import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


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
def distance_score(detection, box):
	detbottom = detection[1] + detection[3]
	detleft = detection[0]
	detright = detection[0] + detection[2]
	boxbottom = box[1] + box[3]
	boxcentre = box[0] + box[2]/2
	xdist = max(detleft-boxcentre, 0, boxcentre - detright)
	dist = calcdistance([0, detbottom], [xdist, boxbottom])
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

def selectbucket(action_dets):
	actionlist = []
	if isinstance(action_dets[0], np.ndarray) is False: # incase onle one action is detected, action_dets must be made to be 2-d
		action_dets = [action_dets]
	for actionline, action in enumerate(action_dets): #go through each action in the video
		found = False
		for currentaction in reversed(actionlist):
			if (currentaction[0] == action[0]) and (int(action[3]) - currentaction[2] < 26): # check if that baboon has done an action recently
				if action[1] in currentaction[1]: # if this actions bucket is already in dict of buckets from recent actions
					currentaction[1][action[1]] = currentaction[1][action[1]] + 1 # then add 1
				else:
					currentaction[1][action[1]] = 1 # or just add to dict set to 1 if it wasnt there before
				currentaction[2] = int(action[3]) # set frame of last action to current frame
				found = True
				break
		if not found: # if this is a new action
			actionlist.append([action[0], {action[1]:1}, int(action[3]), int(action[3])]) # add action to list storing Baboon ID, Dict for possible buckets, and frame number of action
	print(actionlist)
	output_actions = []
	for line in actionlist:
		output_actions.append([])
		output_actions[-1].append(line[0])
		BucketID = '-1'
		maxhits = 0
		for key, value in line[1].items():
			if value >= maxhits:
				BucketID = key # Bucket with the highest total is assigned the action
				maxhits = value
		output_actions[-1].append(BucketID)
		output_actions[-1].append(str(line[3]/25))
	return output_actions




def getbucketnumbers(bucketlist, cap, colab, letter):
	bucketdict = {-1: "-1 n"}
	bucketwait = {}
	if colab:
		from IPython.display import Image
	for framenum, bucketperframe in enumerate(bucketlist):
		for bucket in bucketperframe:
			if bucket[4] not in bucketdict:
				if (bucket[4] not in bucketwait) or (bucketwait[bucket[4]]==0):
					cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
					ret, currentframe = cap.read()
					if colab:
						get_ipython().run_line_magic('matplotlib', 'inline')
					plt.axis("off")
					plt.imshow(cv2.cvtColor(getbucketcrop(bucket, currentframe), cv2.COLOR_BGR2RGB))
					if colab:
						plt.show()
					else:
						plt.show(block=False)
					time.sleep(1)
					bucketnumber = input('Enter bucket number (if obscured press enter to move on): ')
					if bucketnumber != '':
						bucketcolour = input('Enter bucket colour: ')
						bucketdict[bucket[4]] = bucketnumber + ' ' + bucketcolour
					else:
						bucketwait[bucket[4]] = 25
					if not colab:
						plt.close()
				else:
					bucketwait[bucket[4]]=bucketwait[bucket[4]]-1
	#if letter == 'C':
	##C
	#	bucketdict  = {-1: '-1 n', 18: '3 b', 16: '8 b', 14: '3 g', 13: '5 g', 12: '6 g', 11: '10 b', 10: '5 b', 9: '2 b', 8: '2 g', 7: '1 b', 6: '8 g', 5: '1 g', 4: '10 g', 3: '6 b', 2: '4 g', 1: '9 b', 19: '7 g', 17: '7 b', 27: '9 g', 15: '7 b', 45: '10 b', 51: '7 b'}
	##newbucketdict  = {-1: '-1 n', 19: '7 g', 18: '3 b', 17: '6 g', 16: '2 b', 15: '1 g', 14: '10 b', 13: '3 g', 12: '7 b', 11: '8 b', 10: '2 g', 9: '1 b', 8: '8 g', 7: '10 g', 6: '4 b', 5: '5 g', 4: '4 g', 3: '9 b', 2: '6 b', 1: '5 b', 27: '9 g', 72: '7 b', 89: '10 b'}
	#elif letter == 'D':
	## D
	#	bucketdict = {-1: '-1 n', 20: '1 b', 19: '9 g', 18: '10 b', 17: '4 b', 16: '4 g', 15: '3 b', 14: '6 g', 13: '8 g', 12: '7 g', 11: '1 g', 10: '10 g', 9: '8 b', 8: '9 b', 7: '3 g', 6: '2 g', 5: '5 b', 4: '6 b', 3: '5 g', 2: '7 b', 1: '2 b'}
	#elif letter == 'E':
	## E
	#	bucketdict = {-1: '-1 n', 20: '2 b', 19: '7 g', 18: '10 g', 17: '10 b', 16: '8 b', 15: '8 g', 14: '6 b', 13: '2 g', 12: '7 b', 11: '5 g', 10: '6 g', 9: '9 g', 8: '1 g', 7: '3 b', 6: '5 b', 5: '4 g', 4: '3 g', 3: '9 b', 2: '4 b', 1: '1 b'}
	#elif letter == 'F':
	## F
	#	bucketdict = {-1: '-1 n', 20: '3 b', 19: '10 g', 18: '7 g', 17: '6 g', 16: '10 b', 15: '8 g', 14: '4 b', 13: '8 b', 12: '2 b', 11: '6 b', 10: '7 b', 9: '9 g', 8: '9 b', 7: '1 b', 6: '2 g', 5: '1 g', 4: '4 g', 3: '5 g', 2: '3 g', 1: '5 b'}
	#elif letter == 'G':
	## G
	#	bucketdict = {-1: '-1 n', 20: '3 b', 19: '2 g', 18: '4 g', 17: '6 b', 16: '9 b', 15: '1 g', 14: '7 b', 13: '10 g', 12: '3 g', 11: '8 b', 10: '4 b', 9: '10 b', 8: '2 b', 7: '1 b', 6: '5 b', 5: '7 g', 4: '5 g', 3: '6 g', 2: '8 g', 1: '9 g'}
	#elif letter == 'H':
	## H
	#	bucketdict = {-1: '-1 n', 20: '1 b', 19: '10 b', 18: '2 b', 17: '4 g', 16: '5 g', 15: '9 g', 14: '6 b', 13: '4 b', 12: '9 b', 11: '7 g', 10: '3 b', 9: '3 g', 8: '8 g', 7: '1 g', 6: '5 b', 5: '7 b', 4: '6 g', 3: '2 g', 2: '10 g', 1: '8 b', 29: '10 b', 30: '10 g', 50: '10 g', 58: '10 g', 73: '10 g', 88: '7 g', 102: '2 b', 114: '2 b', 126: '7 b', 159: '8 b', 158: '9 b', 170: '7 g', 179: '7 g', 178: '7 b', 193: '7 b', 187: '4 b', 196: '4 g', 202: '2 b'}
	#elif letter == 'I':
	## I
	#	bucketdict = {-1: '-1 n', 20: '1 g', 19: '1 b', 18: '3 b', 17: '10 b', 16: '2 g', 15: '2 b', 14: '10 g', 13: '9 b', 12: '5 g', 11: '4 b', 10: '7 b', 9: '5 b', 8: '3 g', 7: '8 b', 6: '9 g', 5: '4 g', 4: '6 g', 3: '6 b', 2: '7 g', 1: '8 g', 55: '6 g', 59: '6 g', 81: '9 b', 97: '6 g', 111: '7 b', 114: '5 g', 121: '9 b', 133: '9 g', 136: '10 b', 144: '10 b', 169: '9 g'}
	#elif letter == 'J':
	## J
	#	bucketdict = {-1: '-1 n', 20: '8 b', 19: '7 g', 18: '5 g', 17: '4 b', 16: '7 b', 15: '6 b', 14: '8 g', 13: '2 g', 12: '1 b', 11: '10 g', 10: '5 b', 9: '6 g', 8: '2 b', 7: '1 g', 6: '9 b', 5: '3 g', 4: '9 g', 3: '3 b', 2: '4 g', 1: '10 b'}
	#elif letter == 'K':
	## K
	#	bucketdict = {-1: '-1 n', 44: '2 g', 45: '2 g', 32: '5 b', 11: '3 b', 9: '7 g', 38: '8 b', 36: '7 b', 35: '4 g', 34: '6 g', 5: '1 b', 39: '4 b', 20: '6 b', 43: '1 g', 27: '3 g', 26: '9 b', 47: '5 g', 25: '9 g', 6: '2 b', 30: '10 b', 2: '10 g', 42: '8 g'}
	#elif letter == 'B':
	## B
	#	bucketdict = {-1: '-1 n', 20: '2 g', 19: '9 b', 18: '8 b', 17: '1 b', 16: '5 b', 15: '9 g', 14: '8 g', 13: '1 g', 12: '6 b', 11: '7 g', 10: '7 b', 9: '4 g', 8: '6 g', 7: '10 g', 6: '10 b', 5: '4 b', 4: '2 b', 3: '5 g', 2: '3 g', 1: '3 b', 142: '9 b', 224: '6 b', 341: '6 b', 342: '2 g', 350: '3 b', 349: '5 g', 356: '6 b', 357: '6 g', 380: '7 g', 386: '8 b', 392: '8 g', 431: '10 b', 433: '3 g', 436: '10 g', 462: '8 b', 486: '2 g', 542: '9 g', 549: '10 g', 653: '5 g', 713: '5 g', 715: '9 b', 727: '10 b', 725: '8 g', 739: '5 g', 747: '9 b'}
	#elif letter == 'A':
	#	bucketdict = {-1: '-1 n', 17: '10 b', 16: '5 b', 15: '8 b', 14: '8 g', 13: '3 g', 12: '7 g', 11: '4 g', 10: '1 g', 9: '4 b', 8: '6 b', 7: '7 b', 6: '2 b', 5: '3 b', 4: '9 b', 3: '10 g', 1: '2 g', 2: '5 g', 19: '9 g', 18: '6 g', 54: '1 b', 64: '1 b', 84: '5 b', 107: '4 g', 106: '3 b', 105: '1 b', 104: '7 b', 103: '1 g', 102: '2 b', 101: '2 g', 97: '7 g', 186: '1 g', 162: '2 g', 150: '6 g', 154: '4 b', 153: '10 b', 151: '7 b', 159: '3 g', 184: '4 g', 155: '8 g', 189: '1 b', 167: '3 b', 201: '1 b', 200: '5 g', 199: '2 g', 198: '8 g', 204: '2 b', 203: '3 b', 205: '1 g', 210: '9 b', 209: '5 b', 208: '6 b', 207: '7 g', 212: '10 g', 211: '3 g', 213: '8 b', 214: '4 g', 243: '2 g', 241: '8 g', 234: '10 g', 231: '5 g', 225: '1 b', 248: '2 b', 239: '7 b', 250: '5 b', 249: '9 b', 229: '4 b', 230: '3 g', 252: '10 b', 272: '10 b', 259: '7 b', 257: '1 b', 279: '2 b', 278: '10 g', 277: '4 g', 276: '4 b', 275: '3 b', 274: '5 g', 297: '9 g', 296: '8 b', 285: '7 g', 302: '4 b', 301: '10 b', 300: '9 b', 299: '5 b', 298: '6 g', 412: '7 g', 395: '3 g', 344: '10 g', 335: '2 g', 323: '2 b', 315: '5 g', 411: '6 b', 410: '3 b', 400: '5 b', 396: '9 b', 364: '6 g', 341: '10 b', 313: '4 b', 293: '4 g', 418: '9 g', 392: '7 b', 328: '8 b', 346: '8 g', 425: '1 g', 449: '4 g', 436: '9 b', 435: '10 b', 446: '9 g', 459: '7 b', 458: '10 b', 457: '10 g', 456: '8 g', 454: '8 b', 455: '9 b', 441: '6 b', 443: '1 g', 437: '2 b', 430: '4 b', 448: '5 b', 433: '2 g', 444: '3 b', 486: '10 g', 490: '9 g', 489: '8 g', 491: '5 g', 493: '5 b', 492: '6 g', 488: '7 g', 494: '7 b', 502: '2 g', 501: '3 g', 500: '9 b', 499: '6 b', 498: '3 b', 497: '10 b', 496: '4 g', 495: '8 b', 503: '4 b', 504: '2 b', 505: '1 g', 506: '1 b', 555: '8 g', 534: '6 b', 568: '9 b', 567: '5 b', 566: '10 b', 565: '4 b', 564: '1 g', 562: '8 b', 561: '10 g', 560: '5 g', 559: '7 b', 541: '3 b', 511: '5 g', 686: '2 g', 702: '3 b', 691: '3 g', 690: '9 g', 628: '8 b', 610: '2 b', 707: '6 g', 705: '8 g', 714: '4 b', 623: '5 b', 717: '4 g', 716: '7 b', 715: '10 g', 719: '10 b', 718: '9 b', 578: '1 b', 695: '1 g', 704: '7 g', 692: '5 g', 813: '3 b', 812: '2 g', 811: '1 g', 798: '6 b', 821: '2 b', 820: '4 g', 819: '7 g', 818: '7 b', 817: '8 b', 816: '3 g', 815: '4 b', 814: '5 g', 804: '9 g', 797: '9 b', 693: '8 g', 824: '10 b', 823: '10 g', 822: '6 g', 921: '2 b', 971: '6 b', 969: '3 g', 981: '7 b', 980: '8 g', 1196: '7 b', 1224: '8 g', 1233: '8 g', 1247: '7 b', 1248: '8 g', 1250: '8 b', 1263: '8 b', 1266: '8 b'}
	#else:
	#	print('panic')
	return bucketdict



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

