import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


def get_next_img():
    counter = 0
    while True:
        title = str(counter)
        img_file =os.path.join(img_path[0],img_path[1] + "_" + title + img_path[2]) 
        if os.path.isfile(img_file):
            counter = counter + 1
        else:
            break
    return img_file

def display_image(image, title="", cmap=None):
    plt.imshow(image, cmap)
    plt.title(title)
    plt.autoscale(tight=True)

    plt.xticks(np.linspace(0,image.shape[1], 5))
    plt.yticks(np.linspace(0,image.shape[0], 5))
    
    if debug >= 3:
        if not title:
            img_file = get_next_img()
        else:
            img_file = os.path.join(img_path[0],img_path[1] + "_" + title + img_path[2])
        plt.savefig(img_file, dpi=300)
    plt.show()    

    return

def display_images(images, titles=[], cmaps=None):    
    if len(images) != len(titles):
        titles = [""]*len(images)
    if len(images) != len(cmaps):
        cmaps = [None]*len(images)

    fig, axs = plt.subplots(nrows=2,ncols=3, sharex=True, sharey=True)
    fig.suptitle('Lane Detection Steps')
    for ax, img, ttl, cm in zip(axs.flat, images, titles, cmaps):
        ax.imshow(img,cm)
        ax.set_title(ttl)
        ax.set_xlim(0,img.shape[1])
        ax.set_ylim(img.shape[0],0)
        ax.set_xticks(np.linspace(0,img.shape[1], 5))
        ax.set_yticks(np.linspace(0,img.shape[0], 5))

    if debug >= 3:
        plt.savefig(os.path.join(img_path[0],img_path[1] + "_all" + img_path[2]), dpi=300)        
    plt.show()
    
    return

def color_filter_masks(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    masks = dict()
    masks["yellow"] = cv2.inRange(hls, np.array([ 10,  0, 90]), np.array([ 50,255,255]))
    masks["white"]  = cv2.inRange(hls, np.array([  0,190,  0]), np.array([255,255,255]))
    masks["black"]  = cv2.inRange(hls, np.array([  0,  0,  0]), np.array([255,150,150]))
    
    return masks

def color_filter(image, maskings):
    masks = color_filter_masks(image)
    masks_in_use = [masks[m] for m in maskings if m in masks]
    if "custom" in maskings:
        masks_in_use.append(maskings["custom"])

    if len(masks_in_use) == 1:
        mask_fin = masks_in_use[0]
    else:
        mask_fin = cv2.bitwise_or(*masks_in_use)

    masked = cv2.bitwise_and(image, image, mask = mask_fin)    

    return masked

def roi(img,roi_nodes):
    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)

    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([roi_nodes]), ignore_mask_color)

    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img):
    return cv2.Canny(grayscale(img), 50, 120)

def compute_lane_lines(img_ymax, lines, lane_parms):
    rightSlope = list()
    leftSlope = list()
    rightIntercept = list()
    leftIntercept = list()

    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > lane_parms["slope_tolerance"] and x1 > lane_parms["laneCenter"] - 50 :
                rightSlope.append(slope)
                rightIntercept.append(y2 - (slope*x2))
            elif slope < -1*lane_parms["slope_tolerance"] and x1 < lane_parms["laneCenter"] + 50:                 
                leftSlope.append(slope)
                leftIntercept.append(y2 - (slope*x2))    

    leftavgSlope      = np.mean(leftSlope)
    leftavgIntercept  = np.mean(leftIntercept)
    rightavgSlope     = np.mean(rightSlope)
    rightavgIntercept = np.mean(rightIntercept)

    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    # Solving for x in y=mx+b
    yDraw = lane_parms["yDraw_len"]*img_ymax
    left_line_x1  = int((yDraw - leftavgIntercept)/leftavgSlope)
    left_line_x2  = int((img_ymax - leftavgIntercept)/leftavgSlope)
    right_line_x1 = int((yDraw - rightavgIntercept)/rightavgSlope)
    right_line_x2 = int((img_ymax - rightavgIntercept)/rightavgSlope)
    
    return left_line_x1, left_line_x2, right_line_x1, right_line_x2, yDraw

def lane_edge_color(color):
    color_palette = { 0 : [255,0,0] , # red 
                      1 : [0,255,0] , # green
                      2 : [255,255,0] # yellow
                    }
    return color_palette[color]

def draw_lane_lines(img, left_line_x1, left_line_x2, right_line_x1, right_line_x2, yDraw, lColor = 0, rColor = 2):
    #color of the result lines
    
    rightColor=lane_edge_color(rColor)
    leftColor=lane_edge_color(lColor)
    
    pts = np.array([[left_line_x1, int(yDraw)],[left_line_x2, int(img.shape[0])],
                    [right_line_x2, int(img.shape[0])],[right_line_x1, int(yDraw)]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts],(0,0,255))      

    cv2.line(img, (left_line_x1, int(yDraw)), (left_line_x2, int(img.shape[0])), leftColor, 10)
    cv2.line(img, (right_line_x1, int(yDraw)), (right_line_x2, int(img.shape[0])), rightColor, 10)

    return img

def make_hough_lines_image(img, lines):
    if lines is not None:
        dst = cv2.Canny(img, 50, 200, None, 3)
        img_lines = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(img_lines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)    
        return img_lines
    else:
        return False

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def detect_lane_lines(lane_parms, img):
    """
    input img should be the output of a Canny transform.
    """
    lines = hough_lines(img, 1, np.pi/180, 10, 20, 100)
    
    img_empty = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_line_x1, left_line_x2, right_line_x1, right_line_x2, yDraw = compute_lane_lines(img_empty.shape[0], lines, lane_parms)
    
    lColor = 0
    rColor = 2
    if "lcolor" in lane_parms:
        lColor = lane_parms["lcolor"]
    
    if "rcolor" in lane_parms:
        rColor = lane_parms["rcolor"]
        
    img_lines = draw_lane_lines(img_empty, left_line_x1, left_line_x2, right_line_x1, right_line_x2, yDraw, lColor, rColor)
    
    return img_lines, make_hough_lines_image(img, lines)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def weightSum(image_ll, image_org):
    return cv2.addWeighted(image_ll, 1, image_org, 0.8, 0)

def make_save_dir(img_path_org):

    path_test, file_full= os.path.split(os.path.realpath(img_path_org))
    fn = os.path.splitext(file_full)[0]
    counter = 0
    while True:
        img_path_save = os.path.join(path_test,fn + "_" + str(counter))
        if os.path.isdir(img_path_save):
            counter = counter + 1
        else:
            os.makedirs(img_path_save)
            global img_path
            img_path = [None]*3
            img_path[0] = img_path_save
            img_path[1], img_path[2] = os.path.splitext(file_full)
            break
    return

def get_lane_lines(img_parms):
    """
    Main jumping into function
    """
    titles = list()
    images = list()
    cmaps = list()
    
    print("Starting Lane Detection")
    print("************************")
    img_in = mpimg.imread(img_parms["image"])
    
    if debug >= 3:
        make_save_dir(img_parms["image"])
        
    images.append(img_in)
    titles.append("Original")
    cmaps.append(None)
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
    
    print("1/5 - Masking Unnecessary Colors")
    img_filtered = color_filter(img_in, img_parms["maskings"])
    images.append(img_filtered)
    titles.append("Color_Filtered")
    cmaps.append(None)
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
        
    print("2/5 - Isolate region of interest")
    img_roi = roi(img_filtered,img_parms["roi"])
    images.append(img_roi)
    titles.append("Isolated_ROI")
    cmaps.append(None)
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
    
    print("3/5 - Edge Detection")
    img_canny = canny(img_roi)
    images.append(img_canny)
    titles.append("Canny_Edges")
    cmaps.append('gray')
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
    
    print("4/5 - Draw and shade lane lines")
    img_hough, img_hough_lines = detect_lane_lines(img_parms["ll"], img_canny)
    images.append(img_hough_lines)
    titles.append("Hough_Lines")
    cmaps.append(None)
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
    
    print("5/5 - Final composite image")
    img_result = weightSum(img_hough, img_in)
    images.append(img_result)
    titles.append("Final_Composite")
    cmaps.append(None)
    if debug >= 2 : display_image(images[-1],titles[-1],cmaps[-1])
    
    if debug : display_images(images,titles,cmaps)

    return img_hough


def test_image_parms():
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
    img_parms["ll"]["rcolor"] = 1
    img_parms["ll"]["lcolor"] = 0
   
    return img_parms

#--------- START
#---------------
if __name__ == "__main__":
    debug = True
    get_lane_lines(test_image_parms())
    print("**************************")
    print("Finished Lane Detect Debug")
