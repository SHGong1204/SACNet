import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector,  show_result_pyplot

model = init_detector('/hy-tmp/SCAFF/ssp/configs/sspnet/fcos.py', '/hy-tmp/SCAFF/fovea/sspnet/epoch_38.pth', device='cuda:0')

input_dir = '/hy-tmp/SCAFF/visdrone/VisDrone2019-DET-test_dev/images/0000011_00234_d_0000001.jpg'
out_dir = 'results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)



name = input_dir
print ('detecting: ' + name)
img = mmcv.imread(name)
# img_resize = mmcv.imresize(img, (2000, 1500))
result = inference_detector(model, img)
# result = nms_cpu(result, model.CLASSES, 0.1)
result_img = show_result_pyplot(model, img, result, 0.5)
model.show_result(img, result, out_file='results/result.jpg')