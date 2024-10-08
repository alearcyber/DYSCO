import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import Dysco


test_number = 1

IMAGE = cv2.imread(f"Data/TestingDiffDiff/test{test_number}/unobstructed-aligned.png")
EXPECTED = cv2.imread(f"Data/TestingDiffDiff/test{test_number}/expected.png")

def get_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    kps_s, desc_s = zip(*sorted(list(zip(keypoints, descriptors)), key=lambda pt: pt[0].response, reverse=True))
    return kps_s, desc_s


def draw_keypoints(image, kps, r=6, c=(240, 32, 160), thickness=2):
    edited = image.copy() #copy of image to draw on
    for kp in kps:
        point = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(edited, point, radius=r, color=c, thickness=thickness)
    return edited




def strongest_kps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    kps_s, desc_s = zip(*sorted(list(zip(keypoints, descriptors)), key=lambda pt: pt[0].response, reverse=True))

    for kp in kps_s[:300]:
        #plt.plot(*kp.pt, 'ro')
        point = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(image, point, radius=6, color=(240, 32, 160), thickness=2)

    cv2.imshow("", image)
    cv2.waitKey()


def randomly_sample_keypoints():
    image = IMAGE.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    kps_s, desc_s = zip(*sorted(list(zip(keypoints, descriptors)), key=lambda pt: pt[0].response, reverse=True))

    kp_sample = random.sample(keypoints, 500)
    for kp in kp_sample:
        #plt.plot(*kp.pt, 'ro')
        point = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(image, point, radius=6, color=(240, 32, 160), thickness=2)

    cv2.imshow("", image)
    cv2.waitKey()



def strongest_kp_each_region(image, row_stride_count, col_stride_count):
    kps_s, desc_s = get_keypoints(image) #get keypoints in the image
    best_kps = []
    best_desc = []
    rows, cols = IMAGE.shape[0], IMAGE.shape[1]
    row_stride, col_stride = np.linspace(0, rows-1, row_stride_count+1).astype(int), np.linspace(0, cols-1, col_stride_count+1).astype(int)

    #iterate over each section
    for r in range(row_stride_count):
        row_lo, row_hi = row_stride[r], row_stride[r+1]
        #print(f"row: lo:{row_lo}, hi:{row_hi}")
        for c in range(col_stride_count):
            col_lo, col_hi = col_stride[c], col_stride[c+1]
            #print("col:", col_lo, col_hi)

            #get which keypoints has the max response within the window
            for i in range(len(kps_s)):
                kp = kps_s[i]
                #if row_lo <= kp.pt[0] < row_hi and col_lo <= kp.pt[1] < col_hi:
                if row_lo <= kp.pt[1] < row_hi and col_lo <= kp.pt[0] < col_hi:
                    #print("found one")
                    best_kps.append(kp)
                    best_desc.append(desc_s[i])
                    break # can stop early, because they are sorted by strength/response, ergo the first found withing the window is the best.

    #return resulting strongest keypoints
    return best_kps, best_desc


def distance_constraint(kp1, kp2, threshold):
    """A constraint for the constrained brute-force matching routine."""
    return np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < threshold



#MAKE THIS RETURN SOMETHING USABLE
def strongest_match_each_region(image, other, row_stride_count, col_stride_count, threshold=10, lowe_ratio=0.75):
    #match keypoints
    kps_query, desc_query = strongest_kp_each_region(image, row_stride_count=row_stride_count, col_stride_count=col_stride_count)  # get strongest in each region
    kps_other, desc_other = get_keypoints(other)
    matches = Dysco.constrained_bf_matcher(desc_query, kps_query, desc_other, kps_other, constraint=distance_constraint, threshold=threshold, norm_type=cv2.NORM_L2, lowe_ratio=lowe_ratio)

    #store return values
    kps1, desc1 = [], []
    kps2, desc2 = [], []

    #parse through matches for more usable format
    for i in range(len(matches)):
        m = matches[i] #match
        point_query, point_other = kps_query[m.queryIdx].pt, kps_other[m.trainIdx].pt
        descriptor_query, descriptor_other = desc_query[m.queryIdx], desc_other[m.trainIdx]

        kps1.append(point_query)
        kps2.append(point_other)

        desc1.append(descriptor_query)
        desc2.append(descriptor_other)

    #return 4 lists
    return kps1, desc1, kps2, desc2



def draw_strongest_matches2():
    #first strides lengths was 3,5
    kps, desc = strongest_kp_each_region(IMAGE, row_stride_count=6, col_stride_count=10)  # get strongest in each region
    kps_other, desc_other = get_keypoints(EXPECTED)
    matches = Dysco.constrained_bf_matcher(desc, kps, desc_other, kps_other, constraint=distance_constraint, threshold=10, norm_type=cv2.NORM_L2, lowe_ratio=0.75)

    #draw matched keypoints
    out1, out2 = np.copy(IMAGE), np.copy(EXPECTED)
    for i in range(len(matches)):
        m = matches[i]
        point1, point2 = kps[m.queryIdx].pt, kps_other[m.trainIdx].pt
        point1 = int(point1[0] + 0.5), int(point1[1] + 0.5)
        point2 = int(point2[0] + 0.5), int(point2[1] + 0.5)
        #first point drawing
        cv2.circle(out1, point1, radius=6, color=(240, 32, 160), thickness=2)
        cv2.putText(out1, str(i), (point1[0] + 10, point1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #second points drawing
        cv2.circle(out2, point2, radius=6, color=(240, 32, 160), thickness=2)
        cv2.putText(out2, str(i), (point2[0] + 10, point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    #show results
    cv2.imshow("observed", out1)
    cv2.imshow("expected", out2)
    cv2.waitKey()










def draw_strongest_kp_per_region():
    #draws the strongest couple hundred keypoints
    kps, desc = get_keypoints(IMAGE)
    strongest = draw_keypoints(IMAGE, kps[:300])
    cv2.imshow("Strongest", strongest)


    #grab the strongest keypoints IN EACH REGION, draw them separately
    kps, desc = strongest_kp_each_region(IMAGE, row_stride_count=3, col_stride_count=5) #get strongest in each region
    out = draw_keypoints(IMAGE, kps, r=6, c=(240, 32, 160), thickness=2)
    cv2.imshow("Per Region", out)


    #strongtest each region for the expected
    kps, desc = strongest_kp_each_region(EXPECTED, row_stride_count=3, col_stride_count=5)
    out2 = draw_keypoints(EXPECTED, kps, r=6, c=(240, 32, 160), thickness=2)
    cv2.imshow("Per Region, Expected", out2)
    cv2.waitKey()




def test_strongest_match():
    #Find matching keypoints in each region
    kps, desc, kps_other, desc_other = strongest_match_each_region(IMAGE, EXPECTED, row_stride_count=3, col_stride_count=5, lowe_ratio=0.75)

    #Copy images to visualize keypoints by drawing dots.
    out1, out2 = np.copy(IMAGE), np.copy(EXPECTED)

    #iterate over keypoints and draw on images
    id = 1
    for kp, kp_other in zip(kps, kps_other):
        #round location to nearest pixel
        point1 = int(kp[0] + 0.5), int(kp[1] + 0.5)
        point2 = int(kp_other[0] + 0.5), int(kp_other[1] + 0.5)

        #draw on output image
        cv2.circle(out1, point1, radius=6, color=(240, 32, 160), thickness=2)
        cv2.putText(out1, str(id), (point1[0] + 10, point1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(out2, point2, radius=6, color=(240, 32, 160), thickness=2)
        cv2.putText(out2, str(id), (point2[0] + 10, point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #increment keypoint identifier
        id += 1

    #show the images
    cv2.imshow("Observed", out1)
    cv2.imshow("Expected", out2)
    cv2.waitKey()









def main():
    test_strongest_match()


if __name__ == "__main__":
    main()
