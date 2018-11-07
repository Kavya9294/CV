import numpy
from PIL import Image
from numpy import array
import math
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
    global height, width
    img = numpy.zeros((width,height))
    for x in range(4, width-4):
        for y in range(4,height-4):
            # print "imgx[x,y]:  ",imgx[x,y]
            # print "imgy[x,y]: ",imgy[x,y]
            # print "imgx[x,y]**2+imgy[x,y]**2", imgx[x,y]**2+imgy[x,y]**2
            # print "math.sqrt((imgx[x,y]**2+imgy[x,y]**2))", math.sqrt((imgx[x,y]**2+imgy[x,y]**2))
            img[x,y]=math.sqrt((imgx[x,y]**2+imgy[x,y]**2))
            # print "img[x,y]= ",img[x,y]
    return img

def nonMaximaSupression(newImg):
    global height, width
    img = newImg


# Common funtion to display images
def displayImg(imgArr):
    img = Image.fromarray(imgArr)
    img.show()

def main():
    global height, width
    img = Image.open("CVAssImg.jpeg").convert('LA')
    imgArr = array(img)
    imgArr = imgArr[:,:,0]
    width = len(imgArr)
    height = len(imgArr[0])
    # imgArr.setFlags(Write=true)
    newImg = gaussianFilter(imgArr)
    # displayImg(newImg)
    # img = Image.fromarray(newImg)
    # img.save("output.png")
    # img.show()
    imgX = getGx(newImg)
    displayImg(imgX)
    imgY = getGy(newImg)
    displayImg(imgY)
    newImg1 = computeMag(imgX, imgY)
    displayImg(newImg1)
    # img = nonMaximaSupression(newImg)
main()

