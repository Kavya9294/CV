import numpy as np
from PIL import Image
from numpy import array
import math
import cv2
import matplotlib.pyplot as plt
# import cv2
# from skimage.io import imread
width = 0
height = 0
def gaussianFilter(newimg):
    global height, width
    sum = 0
    img = newimg.copy()
    # newImg = [width-8][height-8]
    for x in range(4, width-4):
        for y in range(4,height-4):
            # Calculating for the last and first row
            sum = 0
            p = img[x-3,y-3]
            sum += p
            p = img[x-3,y-2]
            sum += p
            p = img[x-3,y-1]
            sum += p*2
            p = img[x-3,y]
            sum += p*2
            p = img[x-3,y+1]
            sum += p*2
            p = img[x-3,y+2]
            sum += p
            p = img[x-3,y+3]
            sum += p
            p = img[x+3,y-3]
            sum += p
            p = img[x+3,y-2]
            sum += p
            p = img[x+3,y-1]
            sum += p*2
            p = img[x+3,y]
            sum += p*2
            p = img[x+3,y+1]
            sum += p*2
            p = img[x+3,y+2]
            sum += p
            p = img[x+3,y+3]
            sum += p

            # Calculating for the 2nd and last but 2nd row

            p = img[x-2,y-3]
            sum += p
            p = img[x-2,y-2]
            sum += p*2
            p = img[x-2,y-1]
            sum += p*2
            p = img[x-2,y]
            sum += p*4
            p = img[x-2,y+1]
            sum += p*2
            p = img[x-2,y+2]
            sum += p*2
            p = img[x-2,y+3]
            sum += p
            p = img[x+2,y-3]
            sum += p
            p = img[x+2,y-2]
            sum += p*2
            p = img[x+2,y-1]
            sum += p*2
            p = img[x+2,y]
            sum += p*4
            p = img[x+2,y+1]
            sum += p*2
            p = img[x+2,y+2]
            sum += p*2
            p = img[x+2,y+3]
            sum += p

            # Calculating for the 3rd and last but 3rd row

            p = img[x-1,y-3]
            sum += p*2
            p = img[x-1,y-2]
            sum += p*2
            p = img[x-1,y-1]
            sum += p*4
            p = img[x-1,y]
            sum += p*8
            p = img[x-1,y+1]
            sum += p*4
            p = img[x-1,y+2]
            sum += p*2
            p = img[x-1,y+3]
            sum += p*2
            p = img[x+1,y-3]
            sum += p*2
            p = img[x+1,y-2]
            sum += p*2
            p = img[x+1,y-1]
            sum += p*4
            p = img[x+1,y]
            sum += p*8
            p = img[x+1,y+1]
            sum += p*4
            p = img[x+1,y+2]
            sum += p*2
            p = img[x+1,y+3]
            sum += p*2
            
            #Calculating for the middle row

            p = img[x,y-3]
            sum += p*2
            p = img[x,y-2]
            sum += p*4
            p = img[x,y-1]
            sum += p*8
            p = img[x,y]
            sum += p*16
            p = img[x,y+1]
            sum += p*8
            p = img[x,y+2]
            sum += p*4
            p = img[x,y+3]
            # print "p: ",p
            sum += p*2
            
            # print "sum: ",sum
            # print "normalized sum: ", sum/140
            img[x,y] = sum/140
            # newImg[i++][j++] = sum/140
    # print "new img: ",img
    return img

def getGx(newImg):
    sum = 0
    img = newImg.copy()
    for x in range(0,4):
        for y in range(0,4):
            img[x,y] = 0

    for x in range(width-4, width):
        for y in range(height-4, height):
            img[x,y]=0

    for x in range(4, width-4):
        for y in range(4, height-4):
            sum =0
            p = img[x-1,y-1]
            sum -= p
            p = img[x-1,y+1]
            sum += p
            p = img[x, y-1]
            sum -= p
            p = img[x, y+1]
            sum += p
            p = img[x+1,y-1]
            sum -= p
            p = img[x+1,y+1]
            sum += p
            # print "gx sum: ",sum
            # print "gx normalized sum: ", abs(sum)%768
            img[x,y]=abs(sum)%756
    return img

def getGy(newImg):
    global height, width
    sum = 0
    img = newImg.copy()

    for x in range(0,4):
        for y in range(0,4):
            img[x,y] = 0
            # print "array: ", img[x,y]

    for x in range(width-4, width):
        for y in range(height-4, height):
            img[x,y]=0
            # print "arraylast4", img[x,y]

    for x in range(4, width-4):
        for y in range(4, height-4):
            sum =0
            p = img[x-1,y-1]
            sum += p
            p = img[x+1, y-1]
            sum -= p
            p = img[x-1,y]
            sum += p
            p = img[x+1,y]
            sum -= p
            p = img[x-1, y+1]
            sum += p
            p = img[x+1,y+1]
            sum -= p
            # print " gy sum: ",sum
            # print "gy normalized sum: ", abs(sum)%768
            img[x,y]=abs(sum)%756
    return img

# Compute magnitude
def computeMag(imgx, imgy):
    # global height, width
    y, x = imgx.shape
    img = np.zeros((y,x))
    for i in range(1,y):
        for j in range(1,x):
            img[i,j]=math.sqrt((imgx[i,j]**2+imgy[i,j]**2))
    return img

def nonMaximaSupression(newImg, imageTheta):
    sector = 0
    y,x = newImg.shape
    img = np.zeros((y,x))
    for i in range(1,y-1):
        for j in range(1,x-1):
            sector = getSector(imageTheta[i,j])
            switcher = {
                    0: newImg[i,j] if newImg[i,j] == max(newImg[i]) else 0,
                    1: newImg[i,j] if newImg[i,j] == max([newImg[i-1,j+1],newImg[i,j],newImg[i+1,j-1]]) else 0,
                    2: newImg[i,j] if newImg[i,j] == max([newImg[i-1,j],newImg[i,j],newImg[i+1,j]]) else 0,
                    3: newImg[i,j] if newImg[i,j] == max([newImg[i-1,j-1],newImg[i,j],newImg[i+1,j+1]]) else 0
                    }
            img[i,j] = switcher.get(sector,0)
            # print "img[i,j]: ", img[i,j]
    return img

def getSector(theta):
    if -22.5 <= theta <= 22.5 or (180-22.5) <= theta <= (180+22.5):
        return 1
    elif (45-22.5) <= theta <= (45+22.5) or (225-22.5) <= theta <= (225+22.5):
        return 1
    elif (90-22.5) <= theta <= (90+22.5) or (270-22.5) <= theta <= (270+22.5):
        return 2
    else:
        return 3

def getGradientAngles(imgx, imgy):
    y,x = imgx.shape
    imgTheta = np.zeros((y,x))
    for i in range(1,y):
        for j in range(1,x):
            if imgx[i,j]!=0:
                imgTheta[i,j] = math.atan(imgy[i,j]/imgx[i,j])
            else:
                imgTheta[i,j] = 0
            # print "img theta: ",imgTheta[i,j]
    # idx = np.where(imgx ==0,imgx, -1)
    # print "idx: ",idx
    # imgx[idx] = 1E-1
    # img = imgy.all()/imgx.all() if imgx.all()!=0 else 0
    # print "img: ",img
    # imgTheta = np.arctan(img)
    # print "imgTheta", imgTheta
    return imgTheta


# Common funtion to display images
def displayImg(imgArr):
    img = Image.fromarray(imgArr)
    # img.show()
    mgplot = plt.imshow(img)
    plt.show()

def conv2d(filter,image):
    m,n = filter.shape
    # print "img: ",image
    padded_image = np.pad(image, m-1,'constant',constant_values=0)
    # print "padded: ",padded_image
    y,x = padded_image.shape
    y = y-m+1
    x = x-n+1
    norm = np.sum(filter)
    normalize = (1,norm)[norm>1]
    conv_image = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            conv_image[i,j] = np.sum(padded_image[i:i + m, j:j + n] * filter)/normalize
    return conv_image

def getHistogram (img):
    y,x = img.shape
    histogram = np.zeros(256,dtype=int)
    for i in range(1,y):
        for j in range(1,x):
            value = int(img[i,j])
            # print "index: ",value
            histogram[value]=histogram[value]+1
    return histogram

def pTile(numValues, histogram):
    # sortedHist = reversed(sorted(histogram.iterkeys()))
    sum = 0
    print "numValues: ", numValues
    for i in range(255,0, -1):
        if sum >= numValues:
            print "i: ", i
            return i
        else:
            sum += histogram[i]

    return 0

def makeBinImage(threshhold, newImg):
    img = newImg.copy()
    y,x = img.shape
    for i in range(y):
        for j in range(x):
            img[i,j] = 255 if img[i,j] > threshhold else 0
    return img

def main():
    global height, width
    img = Image.open("CVAssImg1.bmp").convert('LA')
    imgArr = array(img)
    imgArr = imgArr[:,:,0]
    image = cv2.imread("CVAssImg2.bmp", cv2.IMREAD_GRAYSCALE)
    # width = len(imgArr)
    # height = len(imgArr[0])
    # imgArr.setFlags(Write=true)
    # newImg = gaussianFilter(imgArr)
    # displayImg(newImg)
    # img = Image.fromarray(newImg)
    # img.save("output.png")
    # img.show()
    # imgX = getGx(newImg)
    # displayImg(imgX)
    # imgY = getGy(newImg)
    # displayImg(imgY)
    # newImg1 = computeMag(imgX, imgY)
    # displayImg(newImg1)
    # img = nonMaximaSupression(newImg)
    gaussMask=np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])
    gaussianImg = conv2d(gaussMask,image)
    # displayImg(gaussianImg)
    pMaskx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    imgGx = conv2d(pMaskx,gaussianImg)
    imgGx = np.where(imgGx > 256, imgGx/3, imgGx)
    displayImg(imgGx)

    pMasky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    imgGy = conv2d(pMasky,gaussianImg)
    imgGy = np.where(imgGy > 256, imgGy/3, imgGy)
    displayImg(imgGy)

    prewittImg = computeMag(imgGx,imgGy)
    prewittImg = np.where(prewittImg > 256, prewittImg/3,prewittImg)
    displayImg(prewittImg)
    
    imgTheta = getGradientAngles(imgGx,imgGy)

    sharpEdgedImg = nonMaximaSupression(prewittImg, imgTheta)
    sharpEdgedImg = np.where(sharpEdgedImg > 255, sharpEdgedImg/3, sharpEdgedImg)
    displayImg(sharpEdgedImg)

    # Calculate Histogram
    histogram = getHistogram(sharpEdgedImg)
    
    # Calculate for p = 10%
    y,x = sharpEdgedImg.shape
    totalPixels = y*x
    print "totalpixels: ", totalPixels
    top_10 = int(0.1*totalPixels)
    pTen = pTile(top_10, histogram)
    print "print_10: ", pTen
    imgBinary10 = makeBinImage(pTen, sharpEdgedImg)
    displayImg(imgBinary10)

    #Calcualte for p = 30%
    top_30 = int(0.3*totalPixels)
    pThirty = pTile(top_30, histogram)
    print "print_30: ", pThirty
    imgBinary30 = makeBinImage(pThirty, sharpEdgedImg)
    displayImg(imgBinary30)

    #Calculate for p = 50%
    top_50 = int(0.5*totalPixels)
    pFifty = pTile(top_50, histogram)
    print "print_50: ", pFifty
    imgBinary50 = makeBinImage(pFifty, sharpEdgedImg)
    displayImg(imgBinary50)
    

main()

