import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))
from torch.autograd import Variable
import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import cv2
import src.config.test_opt as opt
import src.config.train_opt as opt2
from src.pix2pixHD.models import pix2pixHD_model as p2p
from src.semanticsegmentation import SegmentationModel
import imageio

from firebase import firebase
import pyrebase    #library for firebase
import json
import random
#---------------------------------------------------

def getrequest():
    print('Will Print Request\n')
    firebase3 = firebase.FirebaseApplication("https://objproject-cd7af.firebaseio.com", None)
    result3 = firebase3.get('/Request', None)
    #print(result3)

    myusr = {'동민': '3', '명기': '4'}
    mydance = {'학원물': '1', '커플댄스': '2','게임댄스':'3'}

    #print(result3)
    if result3== None:
        print("Error : No Request!, will break \n")
        return None, None, None
    mykey = list(result3.keys())  # Alpha
    realname = mykey[0]
    name = myusr[realname]
    realattr1 = result3[mykey[0]]['Attr1']
    attr1 = myusr[realattr1]
    realattr2 = result3[mykey[0]]['Attr2']
    attr2 = mydance[realattr2]
    print("requestde by ", realname, " ID :", name, " with", realattr1, realattr2, "each ID is", attr1, attr2)
    print(type(attr1), type(attr2))
    return realname, name, attr1, attr2

def updatedb(realname,name , attr1 , attr2):

    config = {

        "apiKey": "AIzaSyBqCUjjzZ5wJ7DrQ5q1kd0EI_up5K2KFxw",
        "authDomain" :"objproject-cd7af.firebaseapp.com",
        "databaseURL": "https://objproject-cd7af.firebaseio.com",
        "projectId": "objproject-cd7af",
        "storageBucket": "objproject-cd7af.appspot.com",
        "messagingSenderId": "1035720506965"

    }

    firebase = pyrebase.initialize_app(config)

    db = firebase.database()
    #Dotransfer(name , attr1 , attr2)                #Transfer 영상 생성 Name 과 Attr 전달


    myurl = uploadmv(name)
    removerequest()
    db = db.child("feed").child('2').update({'feed_contents': 'This is Feed Contents',
                                                       'feed_img_url': myurl,
                                                       'feed_like': random.randrange(1,100),
                                                        'id':'123',
                                                        'name':realname})


def removerequest():
    config = {

        "apiKey": "AIzaSyBqCUjjzZ5wJ7DrQ5q1kd0EI_up5K2KFxw",
        "authDomain": "objproject-cd7af.firebaseapp.com",
        "databaseURL": "https://objproject-cd7af.firebaseio.com",
        "projectId": "objproject-cd7af",
        "storageBucket": "objproject-cd7af.appspot.com",
        "messagingSenderId": "1035720506965"

    }

    firebase = pyrebase.initialize_app(config)

    db = firebase.database()

    db = db.child("Request").remove()



def uploadmv(name):
    config = {
        "apiKey": "AIzaSyBqCUjjzZ5wJ7DrQ5q1kd0EI_up5K2KFxw",  # webkey
        "authDomain": "objproject-cd7af.firebaseapp.com",  # 프로젝트ID
        "databaseURL": "https://objproject-cd7af.firebaseio.com",  # database url
        "storageBucket": "objproject-cd7af.appspot.com"  # storage
    }
    firebase = pyrebase.initialize_app(config)

    # Authentication - 필요하면
    # #auth = firebase.auth()
    # #user = auth.sign_in_with_email_and_password("yourid@gmail.com", "????")

    # #업로드할 파일명
    default = "./"
   # uploadfile = default + name +'/'+ name +'.gif'
    uploadfile = default  + '/movie.gif'


    # #업로드할 파일의 확장자 구하기
    s = os.path.splitext(uploadfile)[1]
    # #업로드할 새로운파일이름
    #Option1 현재시간
    #now = datetime.today().strftime("%Y%m%d_%H%M%S")
    #Option2 User's Name
    now = name
    filename = now + s

    # Upload files to Firebase
    storage = firebase.storage()

    storage.child("videos/" + filename).put(uploadfile)
    fileUrl = storage.child("videos/" + filename).get_url(1)
    # 0은 저장소 위치 1은 다운로드 url 경로이다.
    # 동영상 파일 경로를 알았으니 어디에서든지 참조해서 사용할 수 있다.
    print(fileUrl)  # 업로드한 파일과 다운로드 경로를 database에 저장하자. 그래야 나중에 사용할 수 있다. storage에서 검색은 안된다는 것 같다.
    # save files info in database
    db = firebase.database()
    d = {}
    d[filename] = fileUrl
    data = json.dumps(d)
    # results = db.child("files").push(data)
    results = db.child("files").set(data)
    print("OK")  # Retrieve data - 전체 파일목록을 출력해 보자. 안드로이드앱에서 출려하게 하면 된다.
    db = firebase.database()
    files = db.child("files").get().val()  # 딕셔너리로 반환된다.
    print(files)

    return fileUrl

def listener():
    cond = True
    while cond :
        realname ,name, attr1 ,attr2 =getrequest()
        if name ==None:
            continue
        else:
            updatedb(realname,name,attr1,attr2)
            cond = False

def make_vector_channel(y, n_dims, shape=(512, 512)):
    def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).type(dtype)

        return zeros.scatter(scatter_dim, y_tensor, 1)

    onehot = _to_one_hot(y, n_dims)

    x_repeat = shape[1] // n_dims + 1
    return onehot.repeat(shape[0], x_repeat)[:,:512]

def Dotransfer(name, attr1,attr2):

    number_in_source_video = 2
    intname = int(name)
    if attr1 != None:
        intattr1 = int(attr1)


    result_Arr=[]

    #1. normalizing function will be deprecated
    #2. When execute Transfer func, check body coordinations and resize synthesized image
    #with some portion of original source skeleton image

    segment_model = SegmentationModel.SegmentationModel(encoder_model='resnet101', decoder_model='upernet')
    opt.change_dataroot(attr2)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
                            #New Modify :result_dir to result_dir + name

    model = create_model(opt)

    centerpoint_dir = os.path.join(opt.dataroot, "centor_pt.json")

    #below var will be utilized after implementing other functions
    syn_midpoint = 0
    syn_floor_point = 0
    source_midpoint = 0
    source_floor_point = 0
    label_size = 512


    #========load c/root/PycharmProjects/T1/checkpoints1enter point json file====================
    import json
    with open(centerpoint_dir, "r") as f:
        centerpoint_dict = json.load(f)
    #=======================================================

    image_size = centerpoint_dict['image_size']
                                                #New Modify : All Str(1) to name
    load_background_dir = Path('./data/source/' + name + '/images/')
    load_background_dir.mkdir(exist_ok=True)

    import numpy as np
    p2p.Pix2PixHDModel.setInitFlag(True)
    for dataidx, data in enumerate(tqdm(dataset)):
        minibatch = 1

        if dataidx==312:
            break

        background_RGB = np.zeros(shape=(image_size[0], image_size[1], 3), dtype=np.uint8)
        background = np.zeros(shape=(image_size[0], image_size[1], 3), dtype=np.uint8)

        temp = cv2.imread(str(load_background_dir) + "/{:05}.png".format(dataidx))
        background_RGB[:, :, 2] = temp[:, :, 0]
        background_RGB[:, :, 1] = temp[:, :, 1]
        background_RGB[:, :, 0] = temp[:, :, 2]

        if attr1 == None :
            people_num = 1
        else:
            people_num =2       #attr1 != None 이면 사람은 2명

            for idx in range(opt.people_num):
                img_path = data['path'][idx]
                centerpoint = centerpoint_dict[str(idx)][dataidx]
                onehot_map = Variable(make_vector_channel(torch.tensor([idx + 3]), 10).unsqueeze(0).unsqueeze(0))
                concated_label = torch.cat((Variable(data['label'][idx]), onehot_map.cpu()), dim=1)

                # generated = model.inference(data['label'][idx], data['inst'], init_input)
                # generated = model.inference(data['label'][idx], Variable(torch.ceil(data['inst']*100)), Variable(torch.ceil(data['semantic']*100)))
                generated = model.inference(concated_label, Variable(data['inst']))

                syn_image = util.tensor2im(generated.data[0])
                mask = segment_model.segment(syn_image, save_img_path="", save_result=False)
                # syn_image[mask[:,:,0]==0]=(0,0,0)
                # x,

                start_point = [centerpoint[0] - label_size // 2, centerpoint[1] - label_size // 2]
                # if label range is over in real image region, cut off some label image region
                if centerpoint[0] - label_size // 2 < 0:
                    syn_image = syn_image[:, int(abs(centerpoint[0] - label_size // 2)):]
                    mask = mask[:, int(abs(centerpoint[0] - label_size // 2)):]
                    start_point[0] = 0
                syn_shape = syn_image.shape
                if centerpoint[1] - label_size // 2 < 0:
                    syn_image = syn_image[int(abs(centerpoint[1] - label_size // 2)):, :]
                    mask = mask[int(abs(centerpoint[1] - label_size // 2)):, :]
                    start_point[1] = 0
                syn_shape = syn_image.shape

                if centerpoint[0] + label_size // 2 >= image_size[1]:
                    syn_image = syn_image[:, :int(syn_shape[1] - (centerpoint[0] + label_size // 2 - image_size[1]))]
                    mask = mask[:, :int(syn_shape[1] - (centerpoint[0] + label_size // 2 - image_size[1]))]

                syn_shape = syn_image.shape
                if centerpoint[1] + label_size // 2 >= image_size[0]:
                    syn_image = syn_image[:int(syn_shape[0] - (centerpoint[1] + label_size // 2 - image_size[0])), :]
                    mask = mask[:int(syn_shape[0] - (centerpoint[1] + label_size // 2 - image_size[0])), :]

                np.copyto(background_RGB[int(start_point[1]):int(start_point[1] + syn_image.shape[0]),
                          int(start_point[0]):int(start_point[0] + syn_image.shape[1])], syn_image,
                          where=mask.astype(bool))

            background[:, :, 2] = background_RGB[:, :, 0]
            background[:, :, 1] = background_RGB[:, :, 1]
            background[:, :, 0] = background_RGB[:, :, 2]

            # background[int(start_point[1]):int(start_point[1] + syn_image.shape[0]),
            # int(start_point[0]):int(start_point[0] + syn_image.shape[1])] |= syn_image

            # cv2.imwrite(str(save_result_dir)+"/{:05}.png".format(data),background)
            result_Arr.append(background_RGB)
            # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][0], opt.label_nc)),
            #                       ('synthesized_image', background)])
            # visualizer.save_images(webpage, visuals, img_path)

        imageio.mimsave('./movie.gif', result_Arr)
listener()

"""
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0].detach()))])
    visualizer.save_images(webpage, visuals, img_path)

"""
"""

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][0], opt.label_nc)),
                       ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path'][idx]
    visualizer.save_images(webpage, visuals, img_path)



print(model.getInitFlag())
webpage.save()
torch.cuda.empty_cache()


"""