from sort import *
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def tracker(vidname):
    vidpath = "../../../Baboon-Videos/"
    detpath = "../../../Baboon-Videos/Detections/"
    vidinname = vidname + ".MP4"
    vidoutname = "Tracked/" + vidname + "tracked.mp4"
    
    #cap = cv2.VideoCapture(vidpath + vidinname)
    #
    #fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    #vidout = cv2.VideoWriter(vidpath + vidoutname, fourcc, 25.0, (1920,1080))
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    sort_tracker = Sort(max_age=args.max_age, 
                            min_hits=args.min_hits,
                            iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(detpath + "det" + vidname + ".txt", delimiter=',')
    
    with open("../../../Baboon-Videos/Tracked/" + "sort" + vidname + ".txt", 'w') as out_file:
        print("Processing")
        for frame in range(int(seq_dets[:,0].max())):
            frame += 1 #detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
            classid = seq_dets[seq_dets[:, 0]==frame, 7]
            dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1
            if(total_frames%250 == 0):
                print('Tracked: %.2f seconds or %.3f minutes' % (total_frames/25, total_frames/1500))
    
            start_time = time.time()
            trackers, notfound, notyet = sort_tracker.update(dets, classid)
            cycle_time = time.time() - start_time
            total_time += cycle_time
    
            #if (cap.isOpened()):
            #        ret, currentframe = cap.read()
    #
            #        # if currentframe is read correctly ret is True
            #        if (ret):
            #            for d in trackers:
            #                start = (int(d[0]),int(d[1]))
            #                end = (int(d[2]),int(d[3]))
            #                colour = (255,0,0)
            #                thickness = 5
            #                cv2.rectangle(currentframe, start, end , colour, thickness)
            #                font = cv2.FONT_HERSHEY_SIMPLEX
            #                cv2.putText(currentframe, str(d[4]), start, font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            #            for d in notfound:
            #                start = (int(d[0]),int(d[1]))
            #                end = (int(d[2]),int(d[3]))
            #                colour = (0,255,0)
            #                thickness = 5
            #                cv2.rectangle(currentframe, start, end , colour, thickness)
            #                font = cv2.FONT_HERSHEY_SIMPLEX
            #                cv2.putText(currentframe, str(d[4]), start, font, 3, (0, 0, 255), 2, cv2.LINE_AA)
            #            for d in notyet:
            #                start = (int(d[0]),int(d[1]))
            #                end = (int(d[2]),int(d[3]))
            #                colour = (0,0,255)
            #                thickness = 5
            #                cv2.rectangle(currentframe, start, end , colour, thickness)
            #                font = cv2.FONT_HERSHEY_SIMPLEX
            #                cv2.putText(currentframe, str(d[4]), start, font, 3, (255, 0, 0), 2, cv2.LINE_AA)
    #
            #        vidout.write(currentframe)
    
            for d in trackers:
                print('%d,%d,%d,%d,%d,%d,1,%d,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1],d[5]),file=out_file)
    #cap.release()
    #vidout.release()
    
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    return

if __name__ == '__main__':
    filename = 'Melon.Zucchini_Rep4p3_24.10.2017'
    tracker(filename)