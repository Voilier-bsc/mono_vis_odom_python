# %%
import os 
import cv2
import numpy as np
import math
import sys
import types
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



orb = cv2.ORB_create(
                        nfeatures=1000,
                        scaleFactor=1.2,
                        nlevels=8,
                        edgeThreshold=31,
                        firstLevel=0,
                        WTA_K=2,
                        scoreType=cv2.ORB_FAST_SCORE,
                        patchSize=31,
                        fastThreshold=25,
                        )

args = types.SimpleNamespace()
args.config             = '/home/cordin/Vis_odom_python/configs/bisenetv2_city.py'
args.weight_path        = '/home/cordin/Vis_odom_python/model_final_v2_city.pth'

cfg = set_cfg_from_file(args.config)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


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

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params,minEigThreshold=0.001)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2

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
    
    kp1 = fast.detect(img_1,None)
    # kp1 = orb.detect(img_1,None)

    pts1 = np.array([x.pt for x in kp1], dtype=np.float32)

    pts1, pts2 = featureTracking(img_1, img_2, pts1)


    E, mask = cv2.findEssentialMat(pts1,pts2,focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
    _, R_f, t_f, _ = cv2.recoverPose(E, pts1, pts2, focal = focal, pp = pp)
    
    R_f_seg = R_f
    t_f_seg = t_f

    t_gt = np.zeros((3,1),dtype=np.float64)

    prevImage = img_2
    prev_pts = pts2
    prev_pts_seg = pts2
    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)
    
    for numFrame in range(2, MAX_FRAME):
        filename = '/media/cordin/새 볼륨/rosbag/dataset/sequences/02/image_0/{0:06d}.png'.format(numFrame)
        
        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c,cv2.COLOR_BGR2GRAY)
        
        prev_pts, curr_pts = featureTracking(prevImage, currImage, prev_pts)
        prev_pts_seg, curr_pts_seg = featureTracking(prevImage, currImage, prev_pts_seg)


        E_mat, mask_n = cv2.findEssentialMat(curr_pts, prev_pts, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E_mat, curr_pts, prev_pts, focal = focal, pp = pp)

        E_mat_seg, mask_n_seg = cv2.findEssentialMat(curr_pts_seg, prev_pts_seg, focal = focal, pp = pp, method=cv2.RANSAC, prob = 0.999, threshold=1.0)
        _, R_seg, t_seg, _ = cv2.recoverPose(E_mat_seg, curr_pts_seg, prev_pts_seg, focal = focal, pp = pp)
        
        abs_scale, t_gt = getScale(numFrame, t_gt)
        
        t_f = t_f + abs_scale*R_f.dot(t)
        R_f = R.dot(R_f)

        t_f_seg = t_f_seg + abs_scale*R_f_seg.dot(t_seg)
        R_f_seg = R_seg.dot(R_f_seg)


        if(prev_pts.shape[0] < 1000):
            im = cv2.resize(currImage_c,(640,480))
            im = im[:, :, ::-1]

            im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

            # inference
            out = net(im).squeeze().detach().cpu().numpy()

            pred = np.where(out==0,19,out)
            pred = palette[pred]
            out = encode_labels(out)
            pred_dynamic = palette[out]
            
            base_mask = np.array(out, dtype=np.uint8) 
            base_mask = cv2.resize(base_mask,(1241,376))

            # kp_curr = orb.detect(currImage,None)
            kp_curr = fast.detect(currImage)
            # kp_curr_seg = orb.detect(currImage, base_mask)
            kp_curr_seg = fast.detect(currImage, base_mask)
            feature_img = cv2.drawKeypoints(currImage_c, kp_curr, None)
            feature_img_seg = cv2.drawKeypoints(currImage_c, kp_curr_seg, None)

            curr_pts = np.array([x.pt for x in kp_curr], dtype=np.float32)
            curr_pts_seg = np.array([x.pt for x in kp_curr_seg], dtype=np.float32)
            cv2.imshow("feat_img", feature_img)
            cv2.imshow("feat_img_seg", feature_img_seg)
            cv2.imshow("pred", cv2.resize(pred_dynamic,(1241,376)))
            

        prevImage = currImage
        prev_pts = curr_pts
        prev_pts_seg = curr_pts_seg
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
        

        cv2.rectangle(traj, (10,10), (600,150), (0,0,0), -1)
        text1 = 'orb Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f[0]),float(t_f[1]),float(t_f[2]))
        cv2.putText(traj, text1, textOrg1, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)


        text2 = 'seg Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_f_seg[0]),float(t_f_seg[1]),float(t_f_seg[2]))
        cv2.putText(traj, text2, textOrg2, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)


        text3 = 'gt  Coordinates: x = {0:02f}m y = {1:02f}m z = {2:02f}m'.format(float(t_gt[0]),float(t_gt[1]),float(t_gt[2]))
        cv2.putText(traj, text3, textOrg3, cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1,8)


        cv2.imshow("trajectory", traj)
        # cv2.imshow("img", currImage_c)
        
        
        cv2.waitKey(1)
    
    cv2.imwrite("result_opt.png",traj)

# %%

