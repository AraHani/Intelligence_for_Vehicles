# Intelligence_for_Vehicles
차량지능기초 수업
자율주행 인지에 관련된 Data-Set 
1. BDD100K는 BAIR(Berkeley Artificial Intelligence Research)에서 제작한 주행영상 data set으로KITTI 나 ApolloScape 등과 비교해서, 다양한 도시와 날씨 환경, 주행 시각과 압도적으로 많은 양의 시퀸스와 이미지 양으로 강점을 보이고 있다. 
 
약 10만개의 시퀸스와 1억개가 넘는 이미지로 구성되어 있으며, 도로 객체 감지에 활용할 수 있다. 특히 위와 같이 lane detection을 위한 뚜렷한 정보를 제공해준다. 또한 주행 가능한 구역(차도) 과 주행 불가능 영역(인도)를 다른 색상으로 구분해 인공지능의 운전 가능 지역 학습에도 사용할 수 있다.

2. nuScenes는 자율주행 data-set 중에서도, 차량에 탑재된 라이다 센서와 레이더 등을 이용해 도시 주행 상황을 재현할 수 있다는 특징을 가지고 있다. 20초 길이의 장면 1000개로 구성되어 있으며 보스턴과 싱가폴, 도시 두 곳의 정보를 보여준다. 
 
위와 같이 촬영 각도가 정해져 있는 타 data set과는 달리, nuScenes는 360도 회전 가능한 유동적인 데이터를 제공한다는 점에서 강점을 보인다. 이러한 데이터로 주변 장애물 감지 (+트래킹), 위험 예측과 라이다 데이터 세그멘테이션(세분화) 작업에 사용할 수 있다.

3. Waymo Open Dataset은 구글 자율주행 자회사 웨이모에서 제작한 data set으로, 크게 Motion Dataset과 Perception Dataset으로 나눌 수 있다. 
Motion Dataset의 경우 각 오브젝트를 차량과 보행자, 그리고 자전거로 구분했으며 3D 바운딩 박스 처리가 되어있다. 또한 각 오브젝트마다 예상치 못한 변수 (차선 변경, 예측하지못한 좌 우회전 등등)를 추가했으며, 여기에 미국 일부 도시(LA, 샌 프란시스코, 피닉스 등) 에서의 3D 맵 데이터가 포함되어 있다.
Perception Dataset은 중거리 라이다 1개와 숏 레인지 라이다 4개, 그리고 5개의 카메라로 제작되었으며 각 오브젝트마다 4개의 클래스 (차량, 보행자, 자전거, 표지판) 로 구분했다. 이와 같은 데이터 셋으로 웨이모 에서는 2D와 3D 모두에서 트래킹과 Detection, 그리고 모션 예측 등에 사용할 수 있다고 한다.

-자율주행 인지에 활용되는 오픈 소스-

1. CNN (Convolutional Neural Network)
주로 이미지를 인식하는 데 사용한다. 패턴을 사용해 이미지를 분류하고, 데이터에서 직접 학습한다는 특징을 지닌다.
<1. Feature Extraction>
import numpy as np

from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Densefrom keras.models import Sequential
# Images fed into this model are 512 x 512 pixels with 3 channels

img_shape = (28,28,1)

# Set up the model

model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size

model.add(Conv2D(6,2,input_shape=img_shape))

# Add relu activation to the layer 

model.add(Activation('relu'))

#Pooling

model.add(MaxPool2D(2))

-	입력 데이터를 받아 필터와 커널을 사용해 convolution을 수행하여 피쳐 맵을 생성한다.
<2. Classification>

model.summary

fully connected layer을 1차원 데이터로 변환하는 과정 (flatten 함수 사용)

<3. Training>
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

# dataset with handwritten digits to train the model onfrom keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train,-1)

x_test = np.expand_dims(x_test,-1)

# Train the model, iterating on the data in batches of 32 samples# for 10 epochs

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test,y_test)

백프로파제이션, gradient descend 와 같은 방식으로 동작.

2. YOLO (v4)
Import argparse
	import os
	import platform
	import shutil
	import time
	from pathlib import Path
	

	import cv2
	import torch
	import torch.backends.cudnn as cudnn
	from numpy import random
	

	from models.experimental import attempt_load
	from utils.datasets import LoadStreams, LoadImages
	from utils.general import (
	    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
	from utils.torch_utils import select_device, load_classifier, time_synchronized
	

	from models.models import *
	from models.experimental import *
	from utils.datasets import *
	from utils.general import *
	

	def load_classes(path):
	    # Loads *.names file at 'path'
	    with open(path, 'r') as f:
	        names = f.read().split('\n')
	    return list(filter(None, names))  # filter removes empty strings (such as last line)
	

	def detect(save_img=False):
	    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
	        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
	    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
	

	    # Initialize
	    device = select_device(opt.device)
	    if os.path.exists(out):
	        shutil.rmtree(out)  # delete output folder
	    os.makedirs(out)  # make new output folder
	    half = device.type != 'cpu'  # half precision only supported on CUDA
	

	    # Load model
	    model = Darknet(cfg, imgsz).cuda()
	    try:
	        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
	    except:
	        load_darknet_weights(model, weights[0])
	    #model = attempt_load(weights, map_location=device)  # load FP32 model
	    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	    model.to(device).eval()
	    if half:
	        model.half()  # to FP16
	

	    # Second-stage classifier
	    classify = False
	    if classify:
	        modelc = load_classifier(name='resnet101', n=2)  # initialize
	        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
	        modelc.to(device).eval()
	

	    # Set Dataloader
	    vid_path, vid_writer = None, None
	    if webcam:
	        view_img = True
	        cudnn.benchmark = True  # set True to speed up constant image size inference
	        dataset = LoadStreams(source, img_size=imgsz)
	    else:
	        save_img = True
	        dataset = LoadImages(source, img_size=imgsz)
	

	    # Get names and colors
	    names = load_classes(names)
	    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
	

	    # Run inference
	    t0 = time.time()
	    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	    for path, img, im0s, vid_cap in dataset:
	        img = torch.from_numpy(img).to(device)
	        img = img.half() if half else img.float()  # uint8 to fp16/32
	        img /= 255.0  # 0 - 255 to 0.0 - 1.0
	        if img.ndimension() == 3:
	            img = img.unsqueeze(0)
	

	        # Inference
	        t1 = time_synchronized()
	        pred = model(img, augment=opt.augment)[0]
	

	        # Apply NMS
	        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
	        t2 = time_synchronized()
	

	        # Apply Classifier
	        if classify:
	            pred = apply_classifier(pred, modelc, img, im0s)
	

	        # Process detections
	        for i, det in enumerate(pred):  # detections per image
	            if webcam:  # batch_size >= 1
	                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
	            else:
	                p, s, im0 = path, '', im0s
	

	            save_path = str(Path(out) / Path(p).name)
	            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
	            s += '%gx%g ' % img.shape[2:]  # print string
	            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
	            if det is not None and len(det):
	                # Rescale boxes from img_size to im0 size
	                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
	

	                # Print results
	                for c in det[:, -1].unique():
	                    n = (det[:, -1] == c).sum()  # detections per class
	                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
	

	                # Write results
	                for *xyxy, conf, cls in det:
	                    if save_txt:  # Write to file
	                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
	                        with open(txt_path + '.txt', 'a') as f:
	                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
	

	                    if save_img or view_img:  # Add bbox to image
	                        label = '%s %.2f' % (names[int(cls)], conf)
	                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
	

	            # Print time (inference + NMS)
	            print('%sDone. (%.3fs)' % (s, t2 - t1))
	

	            # Stream results
	            if view_img:
	                cv2.imshow(p, im0)
	                if cv2.waitKey(1) == ord('q'):  # q to quit
	                    raise StopIteration
	

	            # Save results (image with detections)
	            if save_img:
	                if dataset.mode == 'images':
	                    cv2.imwrite(save_path, im0)
	                else:
	                    if vid_path != save_path:  # new video
	                        vid_path = save_path
	                        if isinstance(vid_writer, cv2.VideoWriter):
	                            vid_writer.release()  # release previous video writer
	

	                        fourcc = 'mp4v'  # output video codec
	                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
	                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
	                    vid_writer.write(im0)
	

	    if save_txt or save_img:
	        print('Results saved to %s' % Path(out))
	        if platform == 'darwin' and not opt.update:  # MacOS
	            os.system('open ' + save_path)
	

	    print('Done. (%.3fs)' % (time.time() - t0))
	

	

	if __name__ == '__main__':
	    parser = argparse.ArgumentParser()
	    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
	    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
	    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
	    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	    parser.add_argument('--view-img', action='store_true', help='display results')
	    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	    parser.add_argument('--augment', action='store_true', help='augmented inference')
	    parser.add_argument('--update', action='store_true', help='update all models')
	    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
	    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
	    opt = parser.parse_args()
	    print(opt)
	

	    with torch.no_grad():
	        if opt.update:  # update all models (to fix SourceChangeWarning)
	            for opt.weights in ['']:
	                detect()
	                strip_optimizer(opt.weights)
	        else:
	            detect()







