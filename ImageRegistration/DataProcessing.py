"""
This file will have helpful functions for data mining and whatnot
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2

###################################################
#Helper Function For Plotting The Variances FOR PCA
###################################################
def plot_pca_variance(data, n_components=None):
    # calculate principle components
    pc = PCA(n_components=n_components)
    pc.fit(data)  # try with SCALED data instead?

    # plot explained variance
    plt.bar(range(1, pc.n_components_ + 1), pc.explained_variance_ratio_, align='center', label='Explained Variance')

    # also plot cumulative variance
    cumulative_variance = []
    total = 0
    for i in range(pc.n_components_):
        total += pc.explained_variance_ratio_[i]
        cumulative_variance.append(total)
    plt.step(range(1, pc.n_components_ + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance',
             color='red')

    # clean up and display the plot
    plt.xticks(range(1, pc.n_components_ + 1), range(1, pc.n_components_ + 1))
    for i in range(pc.n_components_):
        text_label = str(round(100 * pc.explained_variance_ratio_[i], 2)) + '%'
        plt.text(i + 1, pc.explained_variance_ratio_[i], text_label, ha='center')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('\'K\'-th Principle Component')
    plt.legend(loc='center right')
    plt.show()


#crops an image
def crop_image(image, mask):
    """ mask is (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = mask  # unpack mask tuple
    image = image[y1:y2, x1:x2]
    return image


#split up an image into n x m,  n is across, m is up and down
def grid_out_image(image, n, m):
    """split the image into n x n sub-images, return in list, ordered like reading order"""
    height, width = image.shape[0], image.shape[1] # extract width and height of the image
    subheight = height//m
    subwidth = width//n

    x_crossections = []
    y_crossections = []

    #declare subimages sampling grid
    subimages = []
    rows, cols = m, n
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(0)
        subimages.append(col)



    for i in range(n):
        x_crossections.append(subwidth * i)

    for i in range(m):
        y_crossections.append(subheight * i)

    for j in range(len(y_crossections)):
        y = y_crossections[j]
        for i in range(len(x_crossections)):
            x = x_crossections[i]
            x1, y1, x2, y2 = x, y, x + subwidth, y + subheight

            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height

            mask = x1, y1, x2, y2
            cropped_section = crop_image(image, mask)
            subimages[j][i] = cropped_section

    return subimages





def test():
    print('Running DataProcessing tests...')
    sampling_grid = grid_out_image(cv2.imread('images/fail3.jpg'), 3, 2)
    for row in range(2):
        for col in range(3):
            cv2.imshow(f'row:{row}, col:{col}', sampling_grid[row][col])
    cv2.waitKey()

if __name__ == '__main__':
    test()
