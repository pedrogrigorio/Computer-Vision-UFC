import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def SIFT(img1, img1_gray, img2, img2_gray):

	# Initiate detector
	sift = cv2.xfeatures2d.SIFT_create()


	# find the keypoints and descriptors

	kp1_s, des1 = sift.detectAndCompute(img1_gray,None)
	kp2_s, des2 = sift.detectAndCompute(img2_gray,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des2,des1,k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	sift_matches = cv2.drawMatches(img2, kp2_s, img1, kp1_s, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	kpsA = np.float32([kp.pt for kp in kp2_s])
	kpsB = np.float32([kp.pt for kp in kp1_s])

	if len(good) > 4:

		src_pts = np.float32([kpsA[m.queryIdx] for m in good])
		dst_pts = np.float32([kpsB[m.trainIdx] for m in good])
		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)

	width = img2.shape[1] + img1.shape[1]
	height = img2.shape[0] + img1.shape[0]

	sift_result = cv2.warpPerspective(img2, H, (width, height))
	sift_result[0:img1.shape[0], 0:img1.shape[1]] = img1

	return sift_matches, sift_result, kp1_s, kp2_s



def KAZE(img1, img1_gray, img2, img2_gray):

	# Initiate detector
	kaze = cv2.KAZE_create()

	# find the keypoints and descriptors

	kp1_k, des1 = kaze.detectAndCompute(img1_gray,None)
	kp2_k, des2 = kaze.detectAndCompute(img2_gray,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des2,des1,k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	kaze_matches = cv2.drawMatches(img2, kp2_k, img1, kp1_k, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	kpsA = np.float32([kp.pt for kp in kp2_k])
	kpsB = np.float32([kp.pt for kp in kp1_k])

	if len(good) > 4:

		src_pts = np.float32([kpsA[m.queryIdx] for m in good])
		dst_pts = np.float32([kpsB[m.trainIdx] for m in good])
		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)

	width = img2.shape[1] + img1.shape[1]
	height = img2.shape[0] + img1.shape[0]

	kaze_result = cv2.warpPerspective(img2, H, (width, height))
	kaze_result[0:img1.shape[0], 0:img1.shape[1]] = img1

	return kaze_matches, kaze_result, kp1_k, kp2_k



def AKAZE(img1, img1_gray, img2, img2_gray):

	# Initiate KAZE detector
	akaze = cv2.AKAZE_create()

	# Find the keypoints and descriptors
	kp1_a, des1 = akaze.detectAndCompute(img1_gray,None)
	kp2_a, des2 = akaze.detectAndCompute(img2_gray,None)

	# BFMatcher with default params
	bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf_akaze.match(des2,des1)

	akaze_matches = cv2.drawMatches(img2, kp2_a, img1, kp1_a, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	kpsA = np.float32([kp.pt for kp in kp2_a])
	kpsB = np.float32([kp.pt for kp in kp1_a])

	if len(matches) > 4:

		src_pts = np.float32([kpsA[m.queryIdx] for m in matches])
		dst_pts = np.float32([kpsB[m.trainIdx] for m in matches])
		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)

	width = img2.shape[1] + img1.shape[1]
	height = img2.shape[0] + img1.shape[0]

	akaze_result = cv2.warpPerspective(img2, H, (width, height))
	akaze_result[0:img1.shape[0], 0:img1.shape[1]] = img1

	return akaze_matches, akaze_result, kp1_a, kp2_a



def ORB(img1, img1_gray, img2, img2_gray):

	# Initiate KAZE detector
	orb = cv2.ORB_create()

	# Find the keypoints and descriptors
	kp1_o, des1 = orb.detectAndCompute(img1_gray,None)
	kp2_o, des2 = orb.detectAndCompute(img2_gray,None)

	# BFMatcher with default params
	bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf_akaze.match(des2,des1)

	orb_matches = cv2.drawMatches(img2, kp2_o, img1, kp1_o, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	kpsA = np.float32([kp.pt for kp in kp2_o])
	kpsB = np.float32([kp.pt for kp in kp1_o])

	if len(matches) > 4:

		src_pts = np.float32([kpsA[m.queryIdx] for m in matches])
		dst_pts = np.float32([kpsB[m.trainIdx] for m in matches])
		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)

	width = img2.shape[1] + img1.shape[1]
	height = img2.shape[0] + img1.shape[0]

	orb_result = cv2.warpPerspective(img2, H, (width, height))
	orb_result[0:img1.shape[0], 0:img1.shape[1]] = img1

	return orb_matches, orb_result, kp1_o, kp2_o



def BRISK(img1, img1_gray, img2, img2_gray):

	# Initiate KAZE detector
	brisk = cv2.BRISK_create()

	# Find the keypoints and descriptors
	kp1_b, des1 = brisk.detectAndCompute(img1_gray,None)
	kp2_b, des2 = brisk.detectAndCompute(img2_gray,None)

	# BFMatcher with default params
	bf_brisk = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf_brisk.match(des2,des1)

	brisk_matches = cv2.drawMatches(img2, kp2_b, img1, kp1_b, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	kpsA = np.float32([kp.pt for kp in kp2_b])
	kpsB = np.float32([kp.pt for kp in kp1_b])

	if len(matches) > 4:

		src_pts = np.float32([kpsA[m.queryIdx] for m in matches])
		dst_pts = np.float32([kpsB[m.trainIdx] for m in matches])
		H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)

	width = img2.shape[1] + img1.shape[1]
	height = img2.shape[0] + img1.shape[0]

	brisk_result = cv2.warpPerspective(img2, H, (width, height))
	brisk_result[0:img1.shape[0], 0:img1.shape[1]] = img1

	return brisk_matches, brisk_result, kp1_b, kp2_b





# Load images
img1 = cv2.imread(sys.argv[1]) # queryImage
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

img2 = cv2.imread(sys.argv[2]) # trainImage
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# Results

sift_matches, sift_result, kp1_s, kp2_s = SIFT(img1, img1_gray, img2, img2_gray)
kaze_matches, kaze_result, kp1_k, kp2_k = KAZE(img1, img1_gray, img2, img2_gray)
akaze_matches, akaze_result, kp1_a, kp2_a = AKAZE(img1, img1_gray, img2, img2_gray)
orb_matches, orb_result, kp1_o, kp2_o = ORB(img1, img1_gray, img2, img2_gray)
brisk_matches, brisk_result, kp1_b, kp2_b = BRISK(img1, img1_gray, img2, img2_gray)

# Plots

fig, axs = plt.subplots(5, 4, figsize=(18,8), constrained_layout=False)
axs[0,0].imshow(cv2.drawKeypoints(img1_gray,kp1_s,None,color=(0,255,0)))
axs[0,0].set_xlabel('SIFT Features'), axs[0,0].set_xticks([]), axs[0,0].set_yticks([])
axs[0,1].imshow(cv2.drawKeypoints(img2_gray,kp2_s,None,color=(0,255,0)))
axs[0,1].set_xlabel('SIFT Features'), axs[0,1].set_xticks([]), axs[0,1].set_yticks([])
axs[0,2].imshow(sift_matches)
axs[0,2].set_xlabel('Matching with SIFT'), axs[0,2].set_xticks([]), axs[0,2].set_yticks([])
axs[0,3].imshow(sift_result)
axs[0,3].set_xlabel('Result'), axs[0,3].set_xticks([]), axs[0,3].set_yticks([])

axs[1,0].imshow(cv2.drawKeypoints(img1_gray,kp1_k,None,color=(0,255,0)))
axs[1,0].set_xlabel('KAZE Features'), axs[1,0].set_xticks([]), axs[1,0].set_yticks([])
axs[1,1].imshow(cv2.drawKeypoints(img2_gray,kp2_k,None,color=(0,255,0)))
axs[1,1].set_xlabel('KAZE Features'), axs[1,1].set_xticks([]), axs[1,1].set_yticks([])
axs[1,2].imshow(kaze_matches)
axs[1,2].set_xlabel('Matching with KAZE'), axs[1,2].set_xticks([]), axs[1,2].set_yticks([])
axs[1,3].imshow(kaze_result)
axs[1,3].set_xlabel('Result'), axs[1,3].set_xticks([]), axs[1,3].set_yticks([])

axs[2,0].imshow(cv2.drawKeypoints(img1_gray,kp1_a,None,color=(0,255,0)))
axs[2,0].set_xlabel('AKAZE Features'), axs[2,0].set_xticks([]), axs[2,0].set_yticks([])
axs[2,1].imshow(cv2.drawKeypoints(img2_gray,kp2_a,None,color=(0,255,0)))
axs[2,1].set_xlabel('AKAZE Features'), axs[2,1].set_xticks([]), axs[2,1].set_yticks([])
axs[2,2].imshow(akaze_matches)
axs[2,2].set_xlabel('Matching with AKAZE'), axs[2,2].set_xticks([]), axs[2,2].set_yticks([])
axs[2,3].imshow(akaze_result)
axs[2,3].set_xlabel('Result'), axs[2,3].set_xticks([]), axs[2,3].set_yticks([])

axs[3,0].imshow(cv2.drawKeypoints(img1_gray,kp1_o,None,color=(0,255,0)))
axs[3,0].set_xlabel('ORB Features'), axs[3,0].set_xticks([]), axs[3,0].set_yticks([])
axs[3,1].imshow(cv2.drawKeypoints(img2_gray,kp2_o,None,color=(0,255,0)))
axs[3,1].set_xlabel('ORB Features'), axs[3,1].set_xticks([]), axs[3,1].set_yticks([])
axs[3,2].imshow(orb_matches)
axs[3,2].set_xlabel('Matching with ORB'), axs[3,2].set_xticks([]), axs[3,2].set_yticks([])
axs[3,3].imshow(orb_result)
axs[3,3].set_xlabel('Result'), axs[3,3].set_xticks([]), axs[3,3].set_yticks([])

axs[4,0].imshow(cv2.drawKeypoints(img1_gray,kp1_b,None,color=(0,255,0)))
axs[4,0].set_xlabel('ORB Features'), axs[4,0].set_xticks([]), axs[4,0].set_yticks([])
axs[4,1].imshow(cv2.drawKeypoints(img2_gray,kp2_b,None,color=(0,255,0)))
axs[4,1].set_xlabel('ORB Features'), axs[4,1].set_xticks([]), axs[4,1].set_yticks([])
axs[4,2].imshow(brisk_matches)
axs[4,2].set_xlabel('Matching with ORB'), axs[4,2].set_xticks([]), axs[4,2].set_yticks([])
axs[4,3].imshow(brisk_result)
axs[4,3].set_xlabel('Result'), axs[4,3].set_xticks([]), axs[4,3].set_yticks([])

plt.show()
