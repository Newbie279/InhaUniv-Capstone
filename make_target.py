import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import os
import warnings
from src.semanticsegmentation import SegmentationModel

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

openpose_dir = Path('./src/PoseEstimation/')

current_mv_name = 4

#=========================
# Time to Complete : 30 minutes
# include openpose running time, +4 minutes
# In this code, make masked images using sematic segmentation

#segment_model = SegmentationModel.SegmentationModel(encoder_model='resnet101', decoder_model='upernet')
#segment_model = SegmentationModel.SegmentationModel(encoder_model='mobilenetv2dilated', decoder_model='c1_deepsup')

def generate_image(current_mv_name):

    save_dir = Path('./data/target/'+str(current_mv_name)+'/')
    save_dir.mkdir(exist_ok=True)

    img_dir = save_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)

    #added model!!!
    #===========
    if len(os.listdir('./data/target/'+str(current_mv_name)+'/images'))<100:
        cap = cv2.VideoCapture(str(save_dir.joinpath('mv'+str(current_mv_name)+'.mp4')))
        i = 0

        while (cap.isOpened()):
            flag, frame = cap.read()
            if flag == False :
                break


            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
            if i%100 == 0:
                print('Has generated %d picetures'%i)
            i += 1



import sys

def make_target(current_mv_name):

    generate_image(current_mv_name=current_mv_name)

    sys.path.append(str(openpose_dir))
    sys.path.append('./src/utils')
    # openpose

    save_dir = Path('./data/target/'+str(current_mv_name)+'/')
    save_dir.mkdir(exist_ok=True)

    img_dir = save_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)

    '''make label images for pix2pix'''
    train_dir = save_dir.joinpath('train')
    train_dir.mkdir(exist_ok=True)

    train_img_dir = train_dir.joinpath('train_img')
    train_img_dir.mkdir(exist_ok=True)
    train_label_dir = train_dir.joinpath('train_label')
    train_label_dir.mkdir(exist_ok=True)
    train_head_dir = train_dir.joinpath('head_img')
    train_head_dir.mkdir(exist_ok=True)

    #changed!!
    train_inst_dir = train_dir.joinpath('train_inst')
    train_inst_dir.mkdir(exist_ok=True)
    train_feat_dir = train_dir.joinpath('train_feat')
    train_feat_dir.mkdir(exist_ok=True)

    train_semantic_dir = train_dir.joinpath('train_semantic')
    train_semantic_dir.mkdir(exist_ok=True)

    train_mic_dir=train_dir.joinpath('train_ss')
    train_mic_dir.mkdir(exist_ok=True)



    for idx in tqdm(range(len(os.listdir(str(img_dir))))):
        img_path = img_dir.joinpath('{:05}.png'.format(idx))
        img = cv2.imread(str(img_path))
        shape_dst = np.min(img.shape[:2])
        oh = (img.shape[0] - shape_dst) // 2
        ow = (img.shape[1] - shape_dst) // 2

        img = img[oh:oh + shape_dst, ow:ow + shape_dst]
        img = cv2.resize(img, (512, 512))

        #mask = segment_model.segment(img, str(train_inst_dir.joinpath('{:05}.png'.format(idx+4829))))

        #mv1
        #background=
        #man=

        #mv2
        #background=202,202,202
        #man=

        #print(mask.shape)
        #cv2.imshow("t",mask)
        #cv2.waitKey(0)
        ###togray


        #img[mask[:,:,0]==0]=(0,0,0)

        """
        semantic_map=mask.copy()
        for idx2 in range(3):
            if idx2==2:
                _, semantic_map[:, :, idx2] = cv2.threshold(semantic_map[:, :, idx2], 45, 255, cv2.THRESH_BINARY)
    
            _, mask[:,:,idx2] = cv2.threshold(mask[:,:,idx2], 45, current_mv_name * (idx2+4) * 10, cv2.THRESH_BINARY)
    
    
    
        mask[mask[:,:,0:3]==0] = current_mv_name*101
        """
        #sometimes python open cv mat type must be converted by np.array
        #pose, coord is head position
        #############################
        cv2.imwrite(str(train_img_dir.joinpath('{:05}.png'.format(idx+45777))), img)
        #cv2.imwrite(str(train_inst_dir.joinpath('{:05}.png'.format(idx))), mask)
        #cv2.imwrite(str(train_semantic_dir.joinpath('{:05}.png'.format(idx))), semantic_map)


make_target(current_mv_name)