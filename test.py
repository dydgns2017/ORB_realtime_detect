# %%
# set module
import cv2, os, glob

# %%
# set target images
files = glob.glob("./images/RGB/*.jpg")
chans = cv2.split(cv2.cvtColor(cv2.imread(files[9]), cv2.COLOR_BGR2RGB))
rgb = ["r", "g", "b"]
for i in range(0, 3):
    cv2.imshow(rgb[i], chans[i])
cv2.waitKey(0)
cv2.destroyAllWindows()