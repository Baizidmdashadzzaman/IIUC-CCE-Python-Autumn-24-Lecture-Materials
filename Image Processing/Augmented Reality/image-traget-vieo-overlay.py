import cv2
import numpy as np

# Initialize webcam capture and video capture
cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('target.jpg')
myVid = cv2.VideoCapture('video.mp4')

# Check if video and image are loaded properly
if not myVid.isOpened() or imgTarget is None:
    print("Error loading video or image.")
    exit()

# Get target image dimensions
heightT, widthT, channelT = imgTarget.shape

# ORB feature detector
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)

# BFMatcher object
bf = cv2.BFMatcher()

while True:
    # Capture frame from webcam
    success, imgWebcam = cap.read()
    if not success:
        print("Error capturing webcam frame.")
        break

    # Read the next frame from video
    successVid, imgVideo = myVid.read()
    if not successVid:
        # If the video ends, loop it from the beginning
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        successVid, imgVideo = myVid.read()

    imgVideo = cv2.resize(imgVideo, (widthT, heightT))

    # Detect keypoints and descriptors in the webcam frame
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)

    # Initialize augmented reality output as the original webcam frame
    imgAug = imgWebcam.copy()

    # Proceed only if descriptors are found in both the target image and the webcam frame
    if des1 is not None and des2 is not None:
        # Perform knn matching if descriptors are found in both images
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Minimum matches needed to find homography
        if len(good) > 10:
            # Get the matched keypoints for homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Find homography matrix to warp the video onto the target image in the webcam
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if matrix is not None:
                h, w, c = imgTarget.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                # Warp the video frame onto the webcam image based on the homography
                imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

                # Create mask for overlay
                maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(maskNew, np.int32(dst), (255, 255, 255))

                maskInv = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgWebcam, imgWebcam, mask=maskInv)
                imgAug = cv2.bitwise_or(imgWarp, imgAug)

    # Display the result: augmented or normal webcam feed
    cv2.imshow('Augmented Reality', imgAug)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
myVid.release()
cv2.destroyAllWindows()
