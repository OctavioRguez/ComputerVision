import numpy as np
import cv2

def imagesColorGray(img, isBuilding):
    imgColor = cv2.imread("./Images/" + img, cv2.IMREAD_COLOR) #Read the color image inside an "Images" folder
    
    #Resize the buildings images to make them viable for edge detection
    if isBuilding:
        resolution = 854, 480
        imgColor = cv2.resize(imgColor, resolution)
    imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY) #Convert the color image to grayscale

    #Show the color and grayscale images
    cv2.imshow("Normal Image", imgColor)
    cv2.imshow("Gray Scale Image", imgGray)
    return [imgColor, imgGray]

def imagesBinary(color, gray):
    #Define thresholds for adjusting the visibility for the color and gray image
    thresholdGray = 128
    thresholdColor = 80
    
    #Select a color channel (Blue, Green, Red)
    b, g, r = cv2.split(color)
    channel = r

    #Make the binarization
    imgBinaryGray = cv2.threshold(gray, thresholdGray, 255, cv2.THRESH_BINARY)[1]
    imgBinaryColor = cv2.threshold(channel, thresholdColor, 255, cv2.THRESH_BINARY)[1]

    #Show the binarized images
    cv2.imshow("Binarized Gray Image", imgBinaryGray)
    cv2.imshow("Binarized Color Image", imgBinaryColor)
    
def noise(gray):
    #Variables for the creation of the gaussian noise
    mean = 3
    std = 15
    levels = [1, 3, 6] #Intensities for the noise

    #Create the noise signal
    noise = np.zeros_like(gray)
    cv2.randn(noise, mean, std)

    #Combine the grayscale image with the noises
    imgNoise1 = cv2.add(gray, noise * levels[0])
    imgNoise2 = cv2.add(gray, noise * levels[1])
    imgNoise3 = cv2.add(gray, noise * levels[2])

    #Show the same image with different intensities of noise
    cv2.imshow("Guassian Noise (First Image)", imgNoise1)
    cv2.imshow("Guassian Noise (Second Image)", imgNoise2)
    cv2.imshow("Guassian Noise (Third Image)", imgNoise3)
    return [imgNoise1, imgNoise2, imgNoise3]

def filters(img1, img2, img3, Filter):
    #Select the corresponding filter to the image with different intensities of noise
    match Filter:
        case "Average":
            #Apply Average filter
            kernel = (3, 3) #Kernel size
            imgFiltered1 = cv2.blur(img1, kernel)
            imgFiltered2 = cv2.blur(img2, kernel)
            imgFiltered3 = cv2.blur(img3, kernel)

        case "Gaussian":
            #Apply Gaussian filter
            kernel = (3, 3) #Kernel size
            imgFiltered1 = cv2.GaussianBlur(img1, kernel, cv2.BORDER_DEFAULT)
            imgFiltered2 = cv2.GaussianBlur(img2, kernel, cv2.BORDER_DEFAULT)
            imgFiltered3 = cv2.GaussianBlur(img3, kernel, cv2.BORDER_DEFAULT)

        case "Median":
            #Apply Median filter
            size = 5 #Kernel size
            imgFiltered1 = cv2.medianBlur(img1, size)
            imgFiltered2 = cv2.medianBlur(img2, size)
            imgFiltered3 = cv2.medianBlur(img3, size)

        case "Binomial":
            #Apply Binomial filter
            coeficients = np.array([1, 2, 1]) #Binomial coeficients
            coeficients = coeficients / np.sum(coeficients) #Adjust the coeficients for the kernel
            imgFiltered1 = cv2.sepFilter2D(img1, -1, coeficients, coeficients)
            imgFiltered2 = cv2.sepFilter2D(img2, -1, coeficients, coeficients)
            imgFiltered3 = cv2.sepFilter2D(img3, -1, coeficients, coeficients)

        case _: #Default case
            print("Filter not available")
    
    #Show the filtered image with different intensities of noise
    cv2.imshow(Filter + " filter (First Image)", imgFiltered1)
    cv2.imshow(Filter + " filter (Second Image)", imgFiltered2)
    cv2.imshow(Filter + " filter (Third Image)", imgFiltered3)

def edgeDetection(gray, Type):
    edges = None
    #Select the corresponding type of algorithm for the edge detection
    match Type:
        case "Canny":
            #Apply canny edge detection
            canny = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(canny, 100, 200, apertureSize = 5)

            #Show the image with the edges
            cv2.imshow("Canny edge detection", edges)
            cv2.waitKey(0)

        case "Hysteresis":
            #Apply Hysteresis Thresholding edge detection
            histeresis = cv2.GaussianBlur(gray, (3,3), 0)
            sobelx = cv2.Sobel(histeresis, cv2.CV_64F, 1, 0, ksize = 3) #Calculate the gradient in the x direction
            sobely = cv2.Sobel(histeresis, cv2.CV_64F, 0, 1, ksize = 3) #Calculate the gradient in the y direction
            mag = cv2.cartToPolar(sobelx, sobely, angleInDegrees = True)[0] #Get the magnitude
            
            #Limits for the magnitude
            low_lim = 50
            up_lim = 150
            #Create the edges
            edges = np.zeros_like(gray)
            edges[(mag >= low_lim) & (mag <= up_lim)] = 255
            edges = cv2.dilate(edges, None) #Apply dilatation to the image
            edges = cv2.erode(edges, None) #Apply erosion to the image
            
            #Show the image with the edges
            cv2.imshow('Hysteresis Thresholding edge detection', edges)
            cv2.waitKey(0)

        case _: #Default case
            print("Algorithm for edge detection not available")
    
    return edges

def morphologicalOperations(edges, Type):
    size = (3, 3) #Kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size) #Create kernel

    #Closing the edges for the Canny algorithm (because the erosion and opening delete most of the edges)
    posEdges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) if Type == "Canny" else edges

    #Erode
    erode = cv2.erode(posEdges, kernel)
    cv2.imshow('Erosion', erode)
    cv2.waitKey(0)     

    #Dilatate
    dilate = cv2.dilate(edges, kernel)
    cv2.imshow('Dilatation', dilate)
    cv2.waitKey(0) 

    #Open
    open = cv2.morphologyEx(posEdges, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opening', open)
    cv2.waitKey(0) 

    #Close
    close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closing', close)
    cv2.waitKey(0)

def detectFigures(edges, frame):
    #Detect circles and lines
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    lines = cv2.HoughLines(edges, 1, theta = np.pi/180, threshold = 100)

    #Show the circles
    if circles is not None: #Start if cirles is different than None
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            #Show the circles in color blue
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)

    #Show circles in the current frame
    if lines is not None: #Start if lines is different than None
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #Show the lines in green
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    #Edges binarization for optimizing the detection for cirles and lines
    thresh = cv2.threshold(edges, 240 , 255, cv2.CHAIN_APPROX_NONE)[1]

    #Find the borders
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    #Show the borders in color red
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    #Show the cirles, lines and other figures
    cv2.imshow('Circles, lineas and polygons', frame)

def exercise1(images):
    #Iterate along the available road signs images
    for image in images:
        [imgColor, imgGray] = imagesColorGray(image, False) #Show the color and grayscale images (get both)
        imagesBinary(imgColor, imgGray) #Binarize those images
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #Available filters: Average , Gaussian, Median, Binomial
        Filter = "Median"

        #Apply Gaussian noise to the grayscale image
        [imgNoise1, imgNoise2, imgNoise3] = noise(imgGray) #Get image with 3 different noise intensities
        filters(imgNoise1, imgNoise2, imgNoise3, Filter) #Apply the selected filter to those images
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def exercise2(images):
    #Iterate along the available building images
    for image in images:
        imgGray = imagesColorGray(image, True)[1] #Show the color and grayscale images (get only the grayscale)

        #Type of systems for edge detection available: Canny, Hysteresis
        Type = "Canny"
        edges = edgeDetection(imgGray, Type) #Apply edge detection (get the edges)
        morphologicalOperations(edges, Type) #Apply morphological operations to the edges recently calculated

        cv2.destroyAllWindows()
    
def exercise3():
    #Start recording
    cam = cv2.VideoCapture(0)
    while True:
        #Press Esc key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

        #Read the current frame from the camera
        frame = cam.read()[1]

        #Convert frame to grayscale and show it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray)

        #Edge detection
        canny = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(canny, 50, 150, apertureSize=3) #Canny edge detection
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #Convert edge image to color
        edges_color[np.where((edges_color == [255, 255, 255]).all(axis=2))] = [0, 255, 255] #Highlight the borders
        
        #Show the edges
        cv2.imshow('Edges', cv2.addWeighted(frame, 0.8, edges_color, 0.2, 0))

        detectFigures(edges, frame)

    #Stop recording
    cam.release()
    cv2.destroyAllWindows()


imagesRoadSigns = ["road13.png", "road32.png", "road47.png", "road81.png", "road120.png", 
          "road151.png", "road167.png", "road194.png", "road312.png", "road339.png"]

imagesBuildings = ["building1.jpg", "building2.jpg", "building3.jpg", "building4.jpg", "building5.jpg"]

#The type of filter can be modified inside the function exercise1() with the variable "Filter" (Default:Average)
#This function uses the following functions: imagesColorGray, imagesBinary, noise, filters.
exercise1(imagesRoadSigns)

#The algorithm for edge detection can be modified inside the function exercise2() with the variable "Type" (Default:Canny)
#This function uses the following functions: imagesColorGray, edgeDetection, morphologicalOperations
exercise2(imagesBuildings)

#This function only uses the detectFigures function to detect the circles, lines and polygons
exercise3()
