import object_detection
import lane_lines
import cv2
import numpy as np

test = 1

if test == 1:
    img_parms = dict()
    img_parms["image"] = "images/solidYellowLeft.jpg"

    # lane line colors
    img_parms["maskings"] = dict()
    img_parms["maskings"]["yellow"] = True
    img_parms["maskings"]["white"] = True
    # lane line region
    img_parms["roi"] = list()
    img_parms["roi"] = np.array([[0,540], [960, 540], [528, 324], [432, 324]])
    # lane lines
    img_parms["ll"] = dict()
    img_parms["ll"]["yDraw_len"] = 0.65
    img_parms["ll"]["slope_tolerance"] = 0.3
    img_parms["ll"]["laneCenter"] = 550 # x pixels
elif test == 2:
    img_parms = dict()
    img_parms["image"] = "images/road_marker.jpg"
    
    # lane line colors
    img_parms["maskings"] = dict()
    img_parms["maskings"]["black"] = True
    # lane line region
    img_parms["roi"] = list()
    img_parms["roi"] = np.array([[0,2989], [2091, 2989], [1881, 298], [209, 298]])
    # lane lines
    img_parms["ll"] = dict()    
    img_parms["ll"]["yDraw_len"] = 0.35
    img_parms["ll"]["slope_tolerance"] = 0.3
    img_parms["ll"]["laneCenter"] = 1000 # x pixels


def weightSum(image_ll, image_org):
    return cv2.addWeighted(image_ll, 0.5, image_org, 1.0, 0)

def object_detection_parms():
    obd_parms = dict()
    obd_parms["yolo"] = "yolo-coco"
    obd_parms["image"] = img_parms["image"]
    obd_parms["confidence"] = 0.5
    obd_parms["threshold"] = 0.3
    return obd_parms
# ---------------
# --_   MAIN ----
# ---------------
if __name__ == "__main__":
    obj_img = object_detection.object_detect(object_detection_parms())
    
    hough_img = lane_lines.get_lane_lines(img_parms)
    
    img_read = cv2.imread(img_parms["image"])
    result_img = weightSum(obj_img, img_read)
    result_img = weightSum(hough_img, result_img)
    
    cv2.imshow("Image", result_img)
    cv2.waitKey(0)
    print("DONE")
