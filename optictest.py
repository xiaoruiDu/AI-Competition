import numpy as np
import cv2
import pickle

DxyvUxy = []
cap = cv2.VideoCapture('slow_traffic_small.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
count = 1
while(1):
    count += 1
    if count == 150:
#        with open('DxyvUxy.pkl','wb') as file:
#            pickle.dump(DxyvUxy,file) 
            break
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(old_gray,cv2.CV_16S,1,0)
    dy = cv2.Sobel(old_gray,cv2.CV_16S,0,1)
    VI = frame_gray - old_gray
    dx = cv2.resize(dx,(6400,3600))
    dy = cv2.resize(dy,(6400,3600))
    VI = cv2.resize(VI,(6400,3600))    

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_old_around = np.around(good_old*10).astype(np.int64)
    for i in range(len(good_old)):
        temp1 = []
        if good_old_around[i][1] >= 3600:
            good_old_around[i][1] = 3599
        if good_old_around[i][0] >= 6400:
            good_old_around[i][0] = 6399
            
        x = dx[(good_old_around[i][1]),(good_old_around[i][0])]
        y = dy[(good_old_around[i][1]),(good_old_around[i][0])]
        vi = (VI[(good_old_around[i][1]),(good_old_around[i][0])])*4
        ux = good_new[i][1] - good_old[i][1]
        uy = good_new[i][0] - good_old[i][0]
        temp1.append(x)
        temp1.append(y)
        temp1.append(vi)
        temp1.append(ux)
        temp1.append(uy)
        DxyvUxy.append(temp1)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('expected flow',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)





cv2.destroyAllWindows()
cap.release()














