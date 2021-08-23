from os import wait
import cv2
import numpy as np
import math
import sys
import types
import operator

## orb 및 bf matcher 선언
orb = cv2.cv2.ORB_create(
                        nfeatures=3000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_FAST_SCORE,
                        patchSize=31,
                        fastThreshold=25,
                        )

bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def getScale(NumFrame, t_gt):

    txt_file = open('/media/cordin/새 볼륨/rosbag/dataset/poses/02.txt')
    
    x_prev = float(t_gt[0])
    y_prev = float(t_gt[1])
    z_prev = float(t_gt[2])

    line = txt_file.readlines()
    line_sp = line[NumFrame].split(' ')

    x = float(line_sp[3])
    y = float(line_sp[7])
    z = float(line_sp[11])

    t_gt[0] = x
    t_gt[1] = y
    t_gt[2] = z

    txt_file.close()

    scale = math.sqrt((x-x_prev)**2 + (y-y_prev)**2 + (z-z_prev)**2)
    return scale, t_gt


if __name__ == "__main__":
    MAX_FRAME = 4541
    
    #Camera intrinsic parameter
    focal = 718.8560
    pp = (607.1928, 185.2157)

    textOrg1 = (10,30)
    textOrg2 = (10,80)
    textOrg3 = (10,130)

    img_1_c = cv2.imread("/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/000000.png")
    img_2_c = cv2.imread("/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/000001.png")

    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img_1,None)
    kp2, des2 = orb.detectAndCompute(img_2,None)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    idx = matches[0:1000]

    pts1 = []
    pts2 = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)


    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    E, mask = cv2.findEssentialMat(pts1,pts2,focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
    _, R_f, t_f, _ = cv2.recoverPose(E, pts1, pts2, focal = focal, pp = pp)

    R_f_seg = R_f
    t_f_seg = t_f

    t_gt = np.zeros((3,1),dtype=np.float64)

    prevImage = img_2
    kp_prev = kp2
    des_prev = des2

    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)

    rmse_total = 0
    
    for numFrame in range(2, 2000):
        filename = '/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/{0:06d}.png'.format(numFrame)
        
        
        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        # feature extraction
        kp_curr, des_curr = orb.detectAndCompute(currImage,None)

        # feature matching
        matches = bf.match(des_prev,des_curr)
        matches = sorted(matches, key = lambda x:x.distance)
        idx = matches[0:3000]

        pts1 = []
        pts2 = []

        for i in idx:
            pts1.append(kp_prev[i.queryIdx].pt)
            pts2.append(kp_curr[i.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        # caculate R, t
        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, pts2, pts1, focal = focal, pp = pp)

        # get scale
        abs_scale, t_gt = getScale(numFrame, t_gt)
        
        # update trajectory
        t_f = t_f + abs_scale*R_f.dot(t)
        R_f = R.dot(R_f)

        # caculate Error
        error = map(operator.sub,t_gt,t_f)
        error_sum_square = sum(map(lambda x:x*x,error))
        rmse = math.sqrt(error_sum_square/3)
        rmse_total = rmse_total + rmse

        print("rmse     = ",rmse_total/numFrame)

        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        # visualization
        x_gt = int(t_gt[0]) + 1000
        y_gt = int(t_gt[2]) + 100

        x = int(t_f[0]) + 1000
        y = int(t_f[2]) + 100

        cv2.circle(traj, (x,y), 1 , (0,0,255), 2)
        cv2.circle(traj, (x_gt,y_gt), 1 , (0,255,0), 2)
        

        cv2.rectangle(traj, (10,10), (700,150), (0,0,0), -1)
        text1 = 'orb Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f[0]),float(t_f[1]),float(t_f[2]))
        cv2.putText(traj, text1, textOrg1, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)

        text3 = 'gt  Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_gt[0]),float(t_gt[1]),float(t_gt[2]))
        cv2.putText(traj, text3, textOrg3, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)

        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)

        cv2.imshow("trajectory", traj)
        cv2.imshow("feat_img", feature_img)

        cv2.waitKey(1)
    
    cv2.imwrite("result_02.png",traj)