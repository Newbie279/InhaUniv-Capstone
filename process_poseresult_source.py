import os
from pathlib import Path
import cv2
from tqdm import tqdm
import json
import math
import operator
import numpy as np

#need to fix

#1. align skeleton to center position(complete)
#2. Time to complete : 20~30 seconds


debug = False
peoples_in_target = 2
current_mv_name = 6
skeleton_ignore_threshold = 0.4
hand_skeleton_ignore_threshold = 0.05
output_size = 512

maximum_diff_between_skeleton=10

source_dir = Path("data/source/"+str(current_mv_name)+"/images/")
source_dir.mkdir(exist_ok=True)

coord_dir = Path("data/source/"+str(current_mv_name)+"/test_coords_ori/")
coord_dir.mkdir(exist_ok=True)

skeleton_dir = Path("./data/source/"+str(current_mv_name)+"/test_label_ori/")
skeleton_dir.mkdir(exist_ok=True)

save_npy_dir = Path('./data/source/'+str(current_mv_name)+'/')
save_npy_dir.mkdir(exist_ok=True)

save_skeleton_dir = Path("data/source/"+str(current_mv_name)+'/test_label/')
save_skeleton_dir.mkdir(exist_ok=True)

save_coord_dir = Path("./data/source/"+str(current_mv_name)+'/')
save_coord_dir.mkdir(exist_ok=True)

#dictionary
man_position_dict = {}
center_point_dict = {}
use_hand_skeleton=False
#return label
#1. want to know label, 2. coord_dict, including all coords in dict
def GetPoseDifference(coord_cur_man, coords_dict):

    next_idx=2

    coord_pos = []
    # get coords
    for idx in range(len(coord_cur_man)):

        if idx == next_idx:  # proba
            if coord_cur_man[idx] > skeleton_ignore_threshold:
                coord_pos.append((coord_cur_man[idx - 2], coord_cur_man[idx - 1]))
            else:
                coord_pos.append((-1, -1))
            next_idx += 3

    max_near_count = 0
    max_count_idx = 0
    non_detect_count = 0
    for key, dict_coord_pos in coords_dict.items():

        temp_near_count = 0
        for idx, pair in enumerate(coord_pos):

            if pair[0] == -1 and pair[1] == -1:
                #panelty
                if dict_coord_pos[idx][0] == -1 and dict_coord_pos[idx][1] == -1:
                    temp_near_count += 1

            else:
                if abs(dict_coord_pos[idx][0] - pair[0]) < maximum_diff_between_skeleton and abs(
                        dict_coord_pos[idx][1] - pair[1]) < maximum_diff_between_skeleton:
                    temp_near_count += 1

        if max_near_count < temp_near_count:
            max_count_idx = key
            max_near_count = temp_near_count

    coords_dict[max_count_idx] = coord_pos
    return max_count_idx


def drawSkeletonMap(shape, coorddata, center_point):

    label = np.zeros(shape, dtype=np.uint8)

    def getCoordDataPair(pt1idx, pt2idx, skeleton_ignore_threshold, get_hand_pos=False, is_left_hand=False):

        if not get_hand_pos:
            coorddataCur = coorddata["pose_keypoints_2d"]
        else:
            if is_left_hand:
                coorddataCur = coorddata["hand_left_keypoints_2d"]


            else:
                coorddataCur = coorddata["hand_right_keypoints_2d"]


        pt1 = [coorddataCur[pt1idx*3], coorddataCur[pt1idx*3+1]]
        pt2 = [coorddataCur[pt2idx*3], coorddataCur[pt2idx*3+1]]


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


    addtional_pos = (shape[0]//2 - center_point[0], shape[1]//2 - center_point[1])

    for idx, pair in enumerate(point_pair):

        pt1, pt2 = getCoordDataPair(pair[0], pair[1], skeleton_ignore_threshold=skeleton_ignore_threshold, get_hand_pos=False)

        if(pt1 == None) or (pt2 == None):
            continue
        pt1[0] += addtional_pos[0]
        pt2[0] += addtional_pos[0]
        pt1[1] += addtional_pos[1]
        pt2[1] += addtional_pos[1]
        draw(pt1, pt2, limb_type=idx, labelMap=label)

    pair = 0
    limb = 24
    #left hand = label 25
    if use_hand_skeleton:

        for idx in range(0,20):

            pt1, pt2 = getCoordDataPair(pair, pair + 1, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=True)

            if ((idx % 4) == 0) and idx != 0:
                pair += 1
                limb += 1
                continue

            if (pt1 == None) or (pt2 == None):
                continue

            pt1[0] += addtional_pos[0]
            pt2[0] += addtional_pos[0]
            pt1[1] += addtional_pos[1]
            pt2[1] += addtional_pos[1]
            draw(pt1, pt2, limb_type=limb, labelMap=label)
            pair += 1

        limb = 24
        for idx2 in range(5, 18, 4):
            pt1, pt2 = getCoordDataPair(0, idx2, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=True)

            limb += 1
            if (pt1 == None) or (pt2 == None):
                continue

            pt1[0] += addtional_pos[0]
            pt2[0] += addtional_pos[0]
            pt1[1] += addtional_pos[1]
            pt2[1] += addtional_pos[1]
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

            pt1[0] += addtional_pos[0]
            pt2[0] += addtional_pos[0]
            pt1[1] += addtional_pos[1]
            pt2[1] += addtional_pos[1]
            draw(pt1, pt2, limb_type=limb, labelMap=label)
            pair += 1

        limb = 24
        for idx2 in range(5, 18, 4):
            pt1, pt2 = getCoordDataPair(0, idx2, skeleton_ignore_threshold=hand_skeleton_ignore_threshold, get_hand_pos=True, is_left_hand=False)

            limb += 1
            if (pt1 == None) or (pt2 == None):
                continue

            pt1[0] += addtional_pos[0]
            pt2[0] += addtional_pos[0]
            pt1[1] += addtional_pos[1]
            pt2[1] += addtional_pos[1]
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
#

def GetCenterPoint(coord_cur_man):

    next_idx=2

    coord_pos = []

    # get coords
    for idx in range(len(coord_cur_man)):

        if idx == next_idx:  # proba
            if coord_cur_man[idx] > skeleton_ignore_threshold:
                coord_pos.append((coord_cur_man[idx - 2], coord_cur_man[idx - 1]))
            else:
                coord_pos.append((-1, -1))
            next_idx += 3

    arr = [0, 0]
    count = 0
    for pair in coord_pos:

        if pair[0] != -1 and pair[1] != -1:
            arr[0] += pair[0]
            arr[1] += pair[1]
            count += 1

    if(count==0):
        return -1,-1,-1

    arr[0] //= count
    arr[1] //= count

    return arr[0], arr[1], coord_pos
"""
    upper = (coord_pos[0][0], coord_pos[0][1])
    lower = ((coord_pos[11][0] + coord_pos[14][0]) / 2, (coord_pos[11][1] + coord_pos[14][1]) / 2)
    right = (coord_pos[9][0], coord_pos[9][1])
    left = (coord_pos[12][0], coord_pos[12][1])
"""
#check all of body coordinations and find the max value in each region
def SetWindow(coord_pos):

    def FindMax(region=""):
        # step=3, Knee, Ankle, Foot
        low_pair = (10, 11, 19, 20, 21)

        # step=3, shoulder, elbow, hand, Ankle
        side_pair = (2, 3, 4, 11)

        # step=0, eye, nose, ear, neck
        high_pair = (0, 1, 15, 16)

        pair = 0
        if region=="top":
            pair=high_pair
        elif region=="side":
            pair=side_pair
        elif region=="low":
            pair=low_pair

        current_max = -1

        if region=="top":
            temp_val = 100000
        else:
            temp_val = 0

        for pair_idx in pair:

            if pair_idx==1 or pair_idx==0:
                if coord_pos[pair_idx][0]!=-1:
                    temp = (coord_pos[pair_idx][0]+coord_pos[pair_idx][1])//2

                    if temp_val > temp:
                        temp_val = temp
                        current_max = (coord_pos[pair_idx][0], coord_pos[pair_idx][1])

            else:
                if region != "side":
                    if coord_pos[pair_idx][0] != -1 and coord_pos[pair_idx+3][0] != -1:
                        temp = ((coord_pos[pair_idx][0]+coord_pos[pair_idx+3][0])//2 + (coord_pos[pair_idx][1]+coord_pos[pair_idx+3][1])//2) // 2

                        if region == "top":
                            if temp_val > temp:
                                temp_val = temp
                                current_max = ((coord_pos[pair_idx][0] + coord_pos[pair_idx + 3][0]) // 2,
                                               (coord_pos[pair_idx][1] + coord_pos[pair_idx + 3][1]) // 2)
                        else:
                            if temp_val < temp:
                                temp_val = temp
                                current_max = ((coord_pos[pair_idx][0] + coord_pos[pair_idx + 3][0]) // 2,
                                               (coord_pos[pair_idx][1] + coord_pos[pair_idx + 3][1]) // 2)

                elif region == "side":
                    if coord_pos[pair_idx][0] != -1 and coord_pos[pair_idx+3][1] != -1:
                        temp = (abs(coord_pos[pair_idx][0]-coord_pos[pair_idx+3][0]) + abs(coord_pos[pair_idx][1]-coord_pos[pair_idx+3][1])) // 2

                        if temp_val < temp:
                            temp_val = temp

                            #[0] is RightPos, [1] is LeftPos
                            current_max = ((coord_pos[pair_idx][0], coord_pos[pair_idx][1]), (coord_pos[pair_idx+3][0], coord_pos[pair_idx+3][1]))

        if region!="side":
            return current_max
        else:
            if current_max==-1:
                return -1, -1
            else:
                return current_max

    upper = FindMax(region="top")
    lower = FindMax(region="low")
    right, left = FindMax(region="side")

    #print("u : "+str(upper)+"l : "+str(lower)+"r : "+str(right)+"l : "+str(left))


    if upper==-1 or lower==-1 or right==-1 or left==-1:
        return (-1, -1, -1, -1)

    x, y, w, h = int(right[0]), int(upper[1]), int((-right[0] + left[0])), int((-upper[1] + lower[1]))

    #print("x : "+str(x)+"y : "+str(y)+"w : "+str(w)+"h : "+str(h))

    return (x, y, w, h)

def GetHist(track_window):

    roi = img[track_window[1]:track_window[1] + track_window[3], track_window[0]:track_window[0] + track_window[2]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    cv2.imshow("t", roi)
    cv2.waitKey(0)

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return roi_hist

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
#determine people number with 1. midpoint  2. from head to botton length 3. diff 4. patience
man_label = {}
track_window = {}

#modify dependancy between skeleton mid position and camshift mid position
midpos_affect_coeff = 0.7
maximum_diff_between_skeleton_and_camshift = 30
init_flag = True
track_patience = 20
for idx in tqdm(range(len(os.listdir(str(skeleton_dir))))):

    #filepath / is needed when imread
    img_path = source_dir.joinpath(str("{:05}".format(idx)+".png"))
    img = cv2.imread(str(img_path))

    #filepath / isn't ndeed when use read
    #find head coordinate
    #coordinations must be subtracted by oh or ow
    coord_path = coord_dir.joinpath(str("videoplayback_{:012}".format(idx)+"_keypoints.json"))
    current_coord = open(str(coord_path))
    coord = json.load(current_coord)

    next_idx = 2

    #1. We assumed this point is midpoint and cropping 512x512 skeleteon pixel
    #check probability of man detection

    proba_result = {}
    for idx_man in range(len(coord["people"])):
        coord_cur_man = coord["people"][idx_man]["pose_keypoints_2d"]

        coord_pos=[]
        avg_proba = 0

        #get coords
        for idx2 in range(len(coord_cur_man)):

            if idx2 == next_idx:#proba

                avg_proba += coord_cur_man[idx2]
                if coord_cur_man[idx2] > skeleton_ignore_threshold:
                    coord_pos.append((coord_cur_man[idx2-2],coord_cur_man[idx2-1]))
                else:
                    coord_pos.append((-1, -1))
                next_idx += 3

        avg_proba /= len(coord_cur_man)//3
        proba_result[idx_man] = avg_proba
        next_idx = 2

        if init_flag:
            man_position_dict[idx_man] = coord_pos
            center_point_dict[idx_man] = []
            print(img.shape)
            center_point_dict['image_size'] = (img.shape[0], img.shape[1])

        """
        # make bounding rect to track people
        if init_flag:

            #make bounding rect to track people

            track_window = SetWindow(coord_pos)
            roi_hist = GetHist(track_window)

            man_label[idx_man] = [track_window, roi_hist, track_patience]

    # use camshift to track the people
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for key in man_label.keys():
        dst = cv2.calcBackProject([hsv], [0], man_label[key][1], [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, man_label[key][0], term_crit)
        #check if track window is to big or odd

        man_label[key][0] = track_window

        if True:
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(img, [pts], True, 100*key, 2)
            cv2.imshow('img2', img2)
            cv2.waitKey(0)
    """

    sorted_proba_result = sorted(proba_result.items(), key=lambda x: x[1], reverse=True)
    #print(sorted_proba_result)
    init_flag=False

    count = 0

    already_used=[]
    for key in sorted_proba_result:

        if count > peoples_in_target:
            break
        coord_cur_man = coord["people"][key[0]]

        pt_x, pt_y, _ = GetCenterPoint(coord_cur_man["pose_keypoints_2d"])

        expected_label = GetPoseDifference(coord_cur_man['pose_keypoints_2d'], man_position_dict)
        #center_point_dict[expected_label].append()
        """
        cpt1, cpt2, coordpos = GetCenterPoint(coord_cur_man["pose_keypoints_2d"])
        min=1000000
        minIdx=-1



        for manIdx, label in man_label.items():

            pt1, pt2 = int((label[0][0]*2 + label[0][2])/2), int((label[0][1]*2 + label[0][3])/2)
            dist = (abs(pt1-cpt1)+abs(pt2-cpt2))

            #1. check the distance between box center and skeleton center
            if min > dist:

                try:
                    already_used.index(manIdx)
                    continue

                except ValueError:
                    minIdx = manIdx
                    min = dist

        already_used.append(manIdx)
        #print(str(manIdx)+','+str(key[0]))
        window = SetWindow(coordpos)
        if window[0] != -1 and idx%update_window_rate==0:
            print(minIdx)
            man_label[minIdx][0] = window
        """
        #resize image and crop, save

        center_point_dict[expected_label].append((pt_x, pt_y))
        img_out=drawSkeletonMap((512, 512), coord_cur_man, (pt_x, pt_y))
        cv2.imwrite(str(save_skeleton_dir.joinpath("p" + str(expected_label) + "/" + str("{:05}".format(idx) + ".png"))),img_out)

#original
#        cv2.imwrite(str(save_skeleton_dir.joinpath("p"+str(expected_label)+"/"+str("{:05}".format(idx) + ".png"))),
#                    drawSkeletonMap((img.shape[0], img.shape[1]), coord_cur_man))
        count += 1

    already_used.clear()
    if count == 0:
        cv2.imwrite(str(save_skeleton_dir.joinpath(str("{:05}".format(idx) + ".png"))),
                    np.zeros(shape=(512, 512), dtype=np.uint8))

with open("./data/source/"+str(current_mv_name)+"/centor_pt.json", "w") as f:
    json.dump(center_point_dict, f)
    f.close()

