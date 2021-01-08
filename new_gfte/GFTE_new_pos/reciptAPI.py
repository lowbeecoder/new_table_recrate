#!coding=utf-8
import requests
import cv2
import numpy as np
# from utils import draw_ocr_box_txt
import json

def draw_bbox(img_path, result, color=(255, 0, 0),thickness=1):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img = img_path.copy()

    for idx,point in enumerate(result):
        point = np.array(point).astype(int)
        cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)
    return img


if __name__=="__main__":
    ## cd sever
    url = r"http://10.17.90.10:9000/predict"
    # url = r"http://10.17.90.10:9000/recipt"

    ## my url
    #url = r"http://127.0.0.1:5000/predict"

    #要预测的文件路径
    # impath = r"D:\OCR\IMAGE\OCR_detexction\test\img\invoice_0396.jpg"
    # impath = r"D:\OCR\IMAGE\OCR_detexction\test\common_recipt\img\invoice_0002.jpg"
    # impath = r"D:\OCR\IMAGE\OCR_detexction\test\common_recipt\img_common\invpic_0066.jpg"
    impath = r"F:\imgs\SciTSR\train\img\0810.1383v2.3.png"


    f = open(impath, 'rb')

    assert f is not None,"no image " + impath

    r = requests.post(url,files = {"file": f})

    output = r.json()

    bblist = output["bbx"]
    labels = output["labellist"]
    print(bblist, labels)
    view = True

    if view:
        im = draw_bbox(impath, bblist)
        cv2.imshow("view",im)
        cv2.waitKey(0)


    # view2 = not view
    # # print(view2)
    # if view2:
    #     imageog = cv2.imread(impath)
    #     im =draw_ocr_box_txt(imageog, bblist, labels)
    #     cv2.imshow("view",im)
    #     cv2.waitKey(0)