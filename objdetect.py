import cv2

img = cv2.imread("datasets/MMW/query/1109_MMW_DJI_0005_00087.jpg", cv2.IMREAD_ANYCOLOR)
img[760:780, 900:920] = 0
cv2.imwrite("test.png", img)


