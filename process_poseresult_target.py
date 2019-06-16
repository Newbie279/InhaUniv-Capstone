import os
from pathlib import Path
import cv2
from tqdm import tqdm
import json
import math
import operator
import numpy as np

#need to fix

#1. align skeleton to center position
#2. Time To Complete : 20s

skeleton_by_original_image=True
original_image_size=1440
debug = False
peoples_in_target = 1
current_mv_name = 4
skeleton_ignore_threshold = 0.33
hand_skeleton_ignore_threshold = 0.05
use_hand_skeleton=False

output_size = 512

def process_poseresult_target(current_mv_name):

    coord_dir = Path("data/target/"+str(current_mv_name)+"/train/train_coord_ori/")
    coord_dir.mkdir(exist_ok=True)

    skeleton_dir = Path("./data/target/"+str(current_mv_name)+"/openpose_res/")
    skeleton_dir.mkdir(exist_ok=True)

    save_npy_dir = Path('./data/target/'+str(current_mv_name)+'/')
    save_npy_dir.mkdir(exist_ok=True)

    save_skeleton_dir = Path("./data/target/"+str(current_mv_name)+'/train/train_label/')
    save_skeleton_dir.mkdir(exist_ok=True)

    save_coord_dir = Path("./data/target/"+str(current_mv_name)+'/')
    save_coord_dir.mkdir(exist_ok=True)


    def drawSkeletonMap(shape, coorddata):

        label = np.zeros(shape, dtype=np.uint8)

        def getCoordDataPair(pt1idx, pt2idx, skeleton_ignore_threshold, get_hand_pos=False, is_left_hand=False):

            if not get_hand_pos:
                coorddataCur = coorddata["pose_keypoints_2d"]
            else:
                if is_left_hand:
                    coorddataCur = coorddata["hand_left_keypoints_2d"]


                else:
                    coorddataCur = coorddata["hand_right_keypoints_2d"]


            pt1 = (coorddataCur[pt1idx*3], coorddataCur[pt1idx*3+1])
            pt2 = (coorddataCur[pt2idx*3], coorddataCur[pt2idx*3+1])


            #if some region moves outside of image 512x512, it will be not drawn
            #if(pt1[0] >= output_size) or (pt2[0] >= output_size) or (pt1[1] >= output_size)

            if(coorddataCur[pt1idx*3+2]>skeleton_ignore_threshold) and (coorddataCur[pt2idx*3+2]>skeleton_ignore_threshold):
                return pt1, pt2
            else:
                return None, None

        def draw(pt1, pt2, limb_type, labelMap):

            coords_center = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))
            limb_dir = (pt1[0] - pt2[0], pt1[1] - pt2[1])
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(labelMap, polygon, limb_type+1)

            if debug:
                cv2.imshow("f",labelMap)
                cv2.waitKey(0)

        point_pair = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),(8,12),
                      (12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(11,22),(22,23),(11,24),(14,19),
                      (19,20),(14,21)]


        for idx, pair in enumerate(point_pair):

            pt1, pt2 = getCoordDataPair(pair[0], pair[1], skeleton_ignore_threshold=skeleton_ignore_threshold, get_hand_pos=False)

            if(pt1 == None) or (pt2 == None):
                continue
            draw(pt1, pt2, limb_type=idx, labelMap=label)
        pair = 0
        limb = 24

        if use_hand_skeleton:
            #left hand = label 25
            for idx in range(0,20):

                pt1, pt2 = getCoordDataPair(pair, pair + 1, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=True)

                if ((idx % 4) == 0) and idx != 0:
                    pair += 1
                    limb += 1
                    continue

                if (pt1 == None) or (pt2 == None):
                    continue
                draw(pt1, pt2, limb_type=limb, labelMap=label)
                pair += 1

            limb = 24
            for idx2 in range(5, 18, 4):
                pt1, pt2 = getCoordDataPair(0, idx2, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=True)

                limb += 1
                if (pt1 == None) or (pt2 == None):
                    continue
                draw(pt1, pt2, limb_type=limb, labelMap=label)

            pair = 0
            limb = 24
            # right hand = label 26
            for idx in range(0, 20):

                pt1, pt2 = getCoordDataPair(pair, pair + 1, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=False)

                if ((idx % 4) == 0) and idx != 0:
                    pair += 1
                    limb += 1
                    continue

                if (pt1 == None) or (pt2 == None):
                    continue
                draw(pt1, pt2, limb_type=limb, labelMap=label)
                pair += 1

            limb = 24
            for idx2 in range(5, 18, 4):
                pt1, pt2 = getCoordDataPair(0, idx2, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=False)

                limb += 1
                if (pt1 == None) or (pt2 == None):
                    continue
                draw(pt1, pt2, limb_type=limb, labelMap=label)

        return label


    #     {0,  "Nose"},
    #     {1,  "Neck"},
    #     {2,  "RShoulder"},
    #     {3,  "RElbow"},
    #     {4,  "RWrist"},
    #     {5,  "LShoulder"},
    #     {6,  "LElbow"},
    #     {7,  "LWrist"},
    #     {8,  "MidHip"},
    #     {9,  "RHip"},
    #     {10, "RKnee"},
    #     {11, "RAnkle"},
    #     {12, "LHip"},
    #     {13, "LKnee"},
    #     {14, "LAnkle"},
    #     {15, "REye"},
    #     {16, "LEye"},
    #     {17, "REar"},
    #     {18, "LEar"},
    #     {19, "LBigToe"},
    #     {20, "LSmallToe"},
    #     {21, "LHeel"},
    #     {22, "RBigToe"},
    #     {23, "RSmallToe"},
    #     {24, "RHeel"},
    #     {25, "Background"}


        #filepath / is needed when imread
    img_path = skeleton_dir.joinpath(str("{:05}".format(0)+".png"))
    img = cv2.imread(str(img_path))

    if debug:
        cv2.imshow("se", img)
        cv2.waitKey(0)

    shape_dst = np.min(img.shape[:2])

    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2
    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    img = cv2.resize(img, (512, 512))


    for idx in tqdm(range(len(os.listdir(str(coord_dir))))):


        #changed

        if debug:
            cv2.imshow("se", img)
            cv2.waitKey(0)


        #filepath / isn't ndeed when use read
        #find head coordinate
        #coordinations must be subtracted by oh or ow
        coord_path = coord_dir.joinpath(str("mv"+str(current_mv_name)+"_{:012}".format(idx)+"_keypoints.json"))
        current_coord = open(str(coord_path))
        coord = json.load(current_coord)

        next_idx = 2

        #check probability of man detection
        proba_result = {}
        if use_hand_skeleton:
            hand_skeleton_len=len(coord["people"][0]["hand_left_keypoints_2d"])

        for idx_man in range(len(coord["people"])):
            coord_cur_man = coord["people"][idx_man]["pose_keypoints_2d"]

            if debug:
                print(coord_cur_man)
                print("length : "+str(len(coord_cur_man)))
                print("index: "+str(idx))

            avg_proba = 0
            for idx2 in range(len(coord_cur_man)):

                if idx2 == next_idx:#proba
                    avg_proba += coord_cur_man[idx2]
                    next_idx += 3
                elif idx2 == (next_idx-2):#width coord
                    coord_cur_man[idx2] -= ow
                    coord_cur_man[idx2] *= output_size/shape_dst

                    if use_hand_skeleton:
                        if idx2 < hand_skeleton_len:
                            coord["people"][idx_man]["hand_left_keypoints_2d"][idx2] -= ow
                            coord["people"][idx_man]["hand_left_keypoints_2d"][idx2] *= output_size/shape_dst

                            coord["people"][idx_man]["hand_right_keypoints_2d"][idx2] -= ow
                            coord["people"][idx_man]["hand_right_keypoints_2d"][idx2] *= output_size/shape_dst

                elif idx2 == (next_idx-1):#height coord
                    coord_cur_man[idx2] -= oh
                    coord_cur_man[idx2] *= output_size/shape_dst

                    if use_hand_skeleton:
                        if idx2 < hand_skeleton_len:
                            coord["people"][idx_man]["hand_left_keypoints_2d"][idx2] -= oh
                            coord["people"][idx_man]["hand_left_keypoints_2d"][idx2] *= output_size/shape_dst

                            coord["people"][idx_man]["hand_right_keypoints_2d"][idx2] -= oh
                            coord["people"][idx_man]["hand_right_keypoints_2d"][idx2] *= output_size/shape_dst

            avg_proba /= len(coord_cur_man)//3
            proba_result[idx_man] = avg_proba
            next_idx = 2


        sorted_proba_result = sorted(proba_result.items(), key=lambda x: x[1], reverse=True)


    #if you want to output multiple skeleton data, change this code!!!!
        count = 0
        for key in sorted_proba_result:

            if count > peoples_in_target:
                break
            coord_cur_man = coord["people"][key[0]]
            cv2.imwrite(str(save_skeleton_dir.joinpath(str("{:05}".format(idx+45777) + ".png"))),
                        drawSkeletonMap((512,512), coord_cur_man))
            count+=1

        if count == 0:
            cv2.imwrite(str(save_skeleton_dir.joinpath(str("{:05}".format(idx+45777) + ".png"))),
                        np.zeros(shape=(512, 512), dtype=np.uint8))

        #pose_cords = np.array(pose_cords, dtype=np.int)
        #np.save(str((save_dir.joinpath('pose.npy'))), pose_cords)



process_poseresult_target(current_mv_name)