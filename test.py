import cv2


orb = cv2.ORB_create(patchSize=100, edgeThreshold=50)
print(orb.getPatchSize()) 
print(orb.getDefaultName())
print(orb.getEdgeThreshold())
img = cv2.imread(".\\images\\Canny\\class_A_0.jpg", 1)
kp, des = orb.detectAndCompute(img, None)
dimg = cv2.drawKeypoints(img, kp, None)
cv2.imshow("dimg", dimg)
cv2.waitKey(0)
cv2.destroyAllWindows()