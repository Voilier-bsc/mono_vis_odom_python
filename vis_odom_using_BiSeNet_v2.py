from os import wait
import cv2
import numpy as np
import math
import sys
import types
import operator


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


import torch
import torch.nn as nn

sys.path.insert(0, '/home/cordin/Vis_odom_python')

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


torch.set_grad_enabled(False)
np.random.seed(123)

mapping = { 
        0: 19,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6, 
        7: 7, 
        8: 8,
        9: 9,
        10: 10,
        11: 0, ## human
        12: 0, ## rider
        13: 0, ## car
        14: 0, ## truck
        15: 0, ## bus
        16: 0, ## train
        17: 0, ## motorcycle
        18: 0, ## bicycle
        -1: 0,
        255: 0
    }

args = types.SimpleNamespace()
args.config             = '/home/cordin/Vis_odom_python/configs/bisenetv2_city.py'
args.weight_path        = '/home/cordin/Vis_odom_python/model_final_v2_city.pth'

cfg = set_cfg_from_file(args.config)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)


# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping:
        label_mask[mask == k] = mapping[k]
    return label_mask

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
    
    focal = 718.8560
    pp = (607.1928, 185.2157)

    textOrg1 = (10,30)
    textOrg2 = (10,80)
    textOrg3 = (10,130)

    img_1_c = cv2.imread("/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/000000.png")
    img_2_c = cv2.imread("/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/000001.png")

    img_1 = cv2.cvtColor(img_1_c,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c,cv2.COLOR_BGR2GRAY)

    # inference
    im = cv2.resize(img_1_c,(640,480))
    im = im[:, :, ::-1]       
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    out = net(im).squeeze().detach().cpu().numpy()

    pred = np.where(out==0,19,out)
    pred = palette[pred]
    out = encode_labels(out)
    pred_dynamic = palette[out]
    
    base_mask = np.array(out, dtype=np.uint8) 
    base_mask = cv2.resize(base_mask,(1241,376))
    ####


    kp1, des1 = orb.detectAndCompute(img_1,None)
    kp2, des2 = orb.detectAndCompute(img_2,None)

    kp1_seg, des1_seg = orb.detectAndCompute(img_1,base_mask)
    kp2_seg, des2_seg = orb.detectAndCompute(img_2,base_mask)


    matches = bf.match(des1,des2)
    matches_seg = bf.match(des1_seg,des2_seg)

    matches = sorted(matches, key = lambda x:x.distance)
    matches_seg = sorted(matches_seg, key = lambda x:x.distance)

    idx = matches[0:1500]
    idx_seg = matches_seg[0:1500]

    pts1 = []
    pts2 = []

    pts1_seg = []
    pts2_seg = []

    for i in idx:
        pts1.append(kp1[i.queryIdx].pt)
        pts2.append(kp2[i.trainIdx].pt)

    for i in idx_seg:
        pts1_seg.append(kp1_seg[i.queryIdx].pt)
        pts2_seg.append(kp2_seg[i.trainIdx].pt)

        
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    pts1_seg = np.array(pts1_seg)
    pts2_seg = np.array(pts2_seg)


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

    kp_prev_seg = kp2_seg
    des_prev_seg = des2_seg

    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)

    rmse_total = 0
    rmse_seg_total = 0

    
    for numFrame in range(2, MAX_FRAME):
        filename = '/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/{0:06d}.png'.format(numFrame)
        currImage_c = cv2.imread(filename)


        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)

        # inference
        im = cv2.resize(currImage_c,(640,480))
        im = im[:, :, ::-1]       
        im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

        out = net(im).squeeze().detach().cpu().numpy()

        pred = np.where(out==0,19,out)
        pred = palette[pred]
        out = encode_labels(out)
        pred_dynamic = palette[out]
        
        base_mask = np.array(out, dtype=np.uint8) 
        base_mask = cv2.resize(base_mask,(1241,376))
        ####

        kp_curr, des_curr = orb.detectAndCompute(currImage,None)
        kp_curr_seg, des_curr_seg = orb.detectAndCompute(currImage,base_mask)

        matches = bf.match(des_prev,des_curr)
        matches_seg = bf.match(des_prev_seg,des_curr_seg)

        matches = sorted(matches, key = lambda x:x.distance)
        matches_seg = sorted(matches_seg, key = lambda x:x.distance)

        idx = matches[0:1500]
        idx_seg = matches_seg[0:1500]
    
        pts1 = []
        pts2 = []

        pts1_seg = []
        pts2_seg = []


        for i in idx:
            pts1.append(kp_prev[i.queryIdx].pt)
            pts2.append(kp_curr[i.trainIdx].pt)


        for j in idx_seg:
            pts1_seg.append(kp_prev_seg[j.queryIdx].pt)
            pts2_seg.append(kp_curr_seg[j.trainIdx].pt)

        
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        pts1_seg = np.array(pts1_seg)
        pts2_seg = np.array(pts2_seg)

        E_mat, mask_n = cv2.findEssentialMat(pts2, pts1, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, pts2, pts1, focal = focal, pp = pp)

        E_seg, mask_seg = cv2.findEssentialMat(pts2_seg, pts1_seg, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R_seg, t_seg, _ = cv2.recoverPose(E_seg, pts2_seg, pts1_seg, focal = focal, pp = pp)
        
        abs_scale, t_gt = getScale(numFrame, t_gt)
        
        t_f = t_f + abs_scale*R_f.dot(t)
        R_f = R.dot(R_f)

        t_f_seg = t_f_seg + abs_scale*R_f_seg.dot(t_seg)
        R_f_seg = R_seg.dot(R_f_seg)


        error = map(operator.sub,t_gt,t_f)
        error_seg = map(operator.sub,t_gt,t_f_seg)

        error_sum_square = sum(map(lambda x:x*x,error))
        error_sum_square_seg = sum(map(lambda x:x*x,error_seg))

        rmse = math.sqrt(error_sum_square/3)
        rmse_seg = math.sqrt(error_sum_square_seg/3)

        rmse_total = rmse_total + rmse
        rmse_seg_total = rmse_seg_total + rmse_seg

        print("rmse     = ",rmse_total/numFrame)
        print("rmse seg = ",rmse_seg_total/numFrame)

        prevImage = currImage
        kp_prev = kp_curr
        des_prev = des_curr

        kp_prev_seg = kp_curr_seg
        des_prev_seg = des_curr_seg

        # x_gt = int(t_gt[0]) + 500
        # y_gt = int(t_gt[2]) + 300

        # x = int(t_f[0]) + 500
        # y = int(t_f[2]) + 300

        # x_seg = int(t_f_seg[0]) + 500
        # y_seg = int(t_f_seg[2]) + 300

        x_gt = int(t_gt[0]) + 1000
        y_gt = int(t_gt[2]) + 100

        x = int(t_f[0]) + 1000
        y = int(t_f[2]) + 100

        x_seg = int(t_f_seg[0]) + 1000
        y_seg = int(t_f_seg[2]) + 100

        cv2.circle(traj, (x,y), 1 , (0,0,255), 2)
        cv2.circle(traj, (x_seg,y_seg), 1 , (255,0,255), 2)
        cv2.circle(traj, (x_gt,y_gt), 1 , (0,255,0), 2)
        

        cv2.rectangle(traj, (10,10), (700,150), (0,0,0), -1)
        text1 = 'orb Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f[0]),float(t_f[1]),float(t_f[2]))
        cv2.putText(traj, text1, textOrg1, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)


        text2 = 'seg Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f_seg[0]),float(t_f_seg[1]),float(t_f_seg[2]))
        cv2.putText(traj, text2, textOrg2, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)


        text3 = 'gt  Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_gt[0]),float(t_gt[1]),float(t_gt[2]))
        cv2.putText(traj, text3, textOrg3, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)

        feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)
        feature_img_seg = cv2.drawKeypoints(currImage_c, kp_curr_seg, None)


        cv2.imshow("trajectory", traj)
        # cv2.imshow("img", currImage_c)
        cv2.imshow("feat_img", feature_img)
        cv2.imshow("feat_img_seg", feature_img_seg)
        cv2.imshow("pred", cv2.resize(pred_dynamic,(1241,376)))


        cv2.waitKey(1)
    
    cv2.imwrite("result_using_BiSeNet_v2.png",traj)