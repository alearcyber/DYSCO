"""
Contains the functionality for separating out an image into a grid.
Implemented as an object
"""
import cv2
import TextureEnergy
import numpy as np
from scipy.stats import chisquare
from scipy import stats
import DataProcessing
from matplotlib import pyplot as plt
import pandas as pd

class ImageGrid:
    def __init__(self, image, rows, cols):
        self.image = image
        self.rows = rows
        self.cols = cols
        self.subimages = []
        self._grid_out_image(image)
        self.dtype = image.dtype

    # split up an image into n x m,  n is across, m is up and down
    def _grid_out_image(self, image):
        """split the image into n x n sub-images, return in list, ordered like reading order"""
        height, width = image.shape[0], image.shape[1]  # extract width and height of the image
        m, n = self.rows, self.cols
        subheight = height // m
        subwidth = width // n

        x_crossections = []
        y_crossections = []

        # declare subimages sampling grid
        #subimages = []
        rows, cols = m, n
        for i in range(rows):
            col = []
            for j in range(cols):
                col.append(0)
            self.subimages.append(col)

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
                cropped_section = self._crop_image(mask)
                self.subimages[j][i] = cropped_section


    # crops an image
    def _crop_image(self, mask):
        """ mask is (x1, y1, x2, y2)"""
        x1, y1, x2, y2 = mask  # unpack mask tuple
        if len(self.image.shape) == 2:
            img = self.image[y1:y2, x1:x2]
        else:
            img = self.image[y1:y2, x1:x2, :]
        return img


    #retreives an individual subimage
    def get(self, row, col):
        return self.subimages[row][col]


    #retrives a mean composite of an individual window
    #exists for convenience in visualization
    def get_composite(self, row, col):
        return np.mean(self.get(row, col), axis=2).astype(np.uint8)




    #for now, just displays all the samples with opencv
    def visualize(self):
        for row in range(self.rows):
            for col in range(self.cols):
                cv2.imshow(f'row:{row}, col:{col}', self.subimages[row][col])
        cv2.waitKey()


    #returns the shape of the individual windows
    def get_subimage_shape(self):
        return self.get(0, 0).shape


    #helper function for constructing mosaic
    #creates single horizontal cross-section for the mosaic
    def _mosaic_cross_section(self, row, partition_width):
        out_shape = (self.get_subimage_shape()[0], 0, 3)
        out = np.empty(shape=out_shape, dtype=self.dtype)
        if len(out.shape) == 2: #convert gray to color
            out = np.expand_dims(out, axis=2)
        vertical_partition = np.array([([[0, 0, 255]] * partition_width)[:] for _ in range(self.get_subimage_shape()[0])], dtype=self.dtype)


        for col in range(self.cols):
            window = self.get(row, col)
            if len(window.shape) == 2:
                window = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)
            if col > 0:  # skip first partition
                out = np.append(out, vertical_partition, axis=1)
            out = np.append(out, window, axis=1)
        return out



    #Creates a Mosaic of the image grid
    def mosaic(self):
        #width of the line separating the different windows
        partition_width = 4

        #Create top row first to avoid extraneous partition lines
        out = self._mosaic_cross_section(0, partition_width)
        horizontal_partition = np.array([([[0, 0, 255]] * partition_width)[:] for _ in range(out.shape[1])], dtype=self.dtype)
        horizontal_partition = np.transpose(horizontal_partition, axes=(1, 0, 2))

        #construct and concatenate rows
        for row in range(1, self.rows):
            cross_section = self._mosaic_cross_section(row, partition_width)
            out = np.append(out, horizontal_partition, axis=0)
            out = np.append(out, cross_section, axis=0)
        return out





    def macro_statistics(self, row, col):
        out = []
        img = self.get(row, col)
        for b in range(img.shape[2]): #iterate over each band
            band = img[:, :, b] #splice out the bth band
            mean = np.mean(band) #grab mean
            out.append(mean) #append mean to output
        return out


    def generate_all_macro_stats(self):
        one_up = []
        for row in range(self.rows):
            stat_row = []
            for col in range(self.cols):
                stats = self.macro_statistics(row, col)
                stat_row.append(stats)
            one_up.append(stat_row)
        return np.array(one_up, dtype=np.uint8)



    def visualize_window_statistics(self):
        return None




def test_visualization():
    image = cv2.imread('images/fail3.jpg')
    #X = TextureEnergy.generate_texture_features(image)
    #X_gray = np.mean(X, axis=2).astype(np.uint8)
    sampling_grid = ImageGrid(image, rows=3, cols=4)
    #sampling_grid.visualize()
    cv2.imshow('test', sampling_grid.get(1, 2))
    cv2.waitKey()


def grab_macro_of_sample():
    #generate stats
    image = cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE)
    X = TextureEnergy.generate_texture_features(image)
    sampling_grid = ImageGrid(X, rows=3, cols=4)
    stats = sampling_grid.macro_statistics(row=1, col=2)
    return stats
    print('stats length:', len(stats))
    print('stats:', stats)

    #visualize band at index 5, 6th band
    #sample = sampling_grid.get(1, 2)[:, :, 5]
    sample = sampling_grid.get(1, 2)[:, :, 0]
    cv2.imshow('sample', sample)
    cv2.waitKey()



def grab_all_macro_stats():
    image = cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE)
    X = TextureEnergy.generate_texture_features(image)
    sampling_grid = ImageGrid(X, rows=3, cols=4)
    stats = sampling_grid.generate_all_macro_stats()
    print(stats)
    #TODO GENERATE THE 3D ARRAY WITH ALL THE MACRO STATISTICS IN ORDER TO DO THE CHI SQUARED TEST
    #TODO CONTINUE HERE



def test_mosaic():
    #image = cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('images/fail3.jpg')
    sampling_grid = ImageGrid(image, rows=3, cols=4)
    cv2.imshow('mosaic', sampling_grid.mosaic())


    composite = sampling_grid.get_composite(row=0, col=0)
    print('composite shape:', composite.shape)
    cv2.imshow('composite', composite)

    cv2.waitKey()


#compare actual image with the other one
def first_chi_squared():
    #read in and align images
    image_expected = cv2.imread("images/rawdisplay.png", cv2.IMREAD_GRAYSCALE)
    image_observed = cv2.imread("images/picture-of-display.png", cv2.IMREAD_GRAYSCALE)
    image_observed = cv2.medianBlur(image_observed, 5)
    image_expected, image_observed = DataProcessing.align_images(image_expected, image_observed)


    #blur
    #image_expected = cv2.GaussianBlur(image_expected, (5,5), 0)


    #morph the texture energy into np array
    texture_expected = TextureEnergy.texture_features9_2(image_expected)
    t = []
    for key in texture_expected:
        t.append(texture_expected[key])
    t = np.array(t, dtype=np.uint8)
    texture_expected = np.transpose(t, axes=(1,2,0))

    texture_observed = TextureEnergy.texture_features9_2(image_observed)
    t = []
    for key in texture_observed:
        t.append(texture_observed[key])
    t = np.array(t, dtype=np.uint8)
    texture_observed = np.transpose(t, axes=(1, 2, 0))


    #create the image grids
    grid_expected = ImageGrid(texture_expected, rows=5, cols=10)
    grid_observed = ImageGrid(texture_observed, rows=5, cols=10)






    #cv2.imshow('composite', np.mean(texture_observed, axis=2).astype(np.uint8))
    cv2.imshow('Observed Mosaic', ImageGrid(image_observed, rows=5, cols=10).mosaic())
    cv2.waitKey()


    observed_window = grid_observed.get(0, 0)
    expected_window = grid_expected.get(0, 0)

    plt.hist(expected_window.ravel(), 50, [0, 256])
    plt.show()
    plt.hist(observed_window.ravel(), 50, [0, 256])
    plt.show()

    assert observed_window.shape[2] == expected_window.shape[2], "different number of bands in observed and expected"

    observed_stats = []
    observed_variance = []
    for i in range(observed_window.shape[2]):
        band = observed_window[:, :, i]
        observed_stats.append(int(np.mean(band)))
        observed_variance.append(int(np.var(band)))


    expected_stats = []
    expected_variance = []
    for i in range(expected_window.shape[2]):
        band = expected_window[:, :, i]
        expected_stats.append(int(np.mean(band)))
        expected_variance.append(int(np.var(band)))

    #remove zeros, make them one?
    for i in range(len(expected_stats)):
        if expected_stats[i] == 0:
            expected_stats[i] = 1

        if observed_stats[i] == 0:
            observed_stats[i] = 1

    observed_stats = np.array(observed_stats, dtype=np.float64)
    expected_stats = np.array(expected_stats, dtype=np.float64)


    print("observed mean:", observed_stats)
    print("expected mean:", expected_stats)
    print()
    print("observed variance:", observed_variance)
    print("expected variance:", expected_variance)


    cv2.imshow("observed window", np.mean(observed_window, axis=2).astype(np.uint8))
    cv2.imshow("expected_window", np.mean(expected_window, axis=2).astype(np.uint8))



    #normalize so data is proportional
    if sum(expected_stats) > sum(observed_stats):
        observed_stats = (observed_stats / sum(observed_stats)) * sum(expected_stats)
    else:
        expected_stats = (expected_stats / sum(expected_stats)) * sum(observed_stats)

    print(observed_stats)
    print(expected_stats)
    #chi squared
    #chi = chisquare(observed_stats, expected_stats)
    #print(chi)

    #t-test
    #NOTE: REL IS FOR COMPARING THE ORDER STATS BETWEEN WINDOWS, IND WOULD BE FOR ALL THE PIXELS VALUES
    ttest = stats.ttest_rel(observed_stats, expected_stats)
    print("related t-test:", ttest)
    ttest = stats.ttest_ind(observed_window.ravel(), expected_window.ravel(), equal_var=False)
    print("independent t-test:", ttest)

    shapriowilk = stats.shapiro(observed_window.ravel())
    print("shapiro:", shapriowilk)

    print("chi-squared distance:", DataProcessing.chi_squared_distance(observed_stats, expected_stats))
    cv2.waitKey()


def chi_squared(a, b):
    print('Chi Squared')
    print(a)
    print(b)





#I want to create a grid where I have the stats for each partition in the grid
# So mean, std, skew, kurtosis, and entropy
def grab_stats_and_visualize():
    # read in and align images
    image_expected = cv2.imread("images/rawdisplay.png", cv2.IMREAD_GRAYSCALE)
    #image_expected = cv2.medianBlur(image_expected, 5)
    image_observed = cv2.imread("images/picture-of-display.png", cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("before", image_observed)
    #cv2.imshow("after", image_observed)
    image_expected, image_observed = DataProcessing.align_images(image_expected, image_observed)
    _, fail = DataProcessing.align_images(image_expected, cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE))
    _, expected2 = DataProcessing.align_images(image_expected, cv2.imread('images/picture1.png', cv2.IMREAD_GRAYSCALE))
    #expected2 = cv2.medianBlur(expected2, 5)


    cv2.imshow('observed', image_observed)
    cv2.imshow('expected', image_expected)
    cv2.waitKey()


    #partition the image
    grid = ImageGrid(image_observed, rows=5, cols=10)
    grid_expected = ImageGrid(image_expected, rows=5, cols=10)
    grid_fail = ImageGrid(fail, rows=5, cols=10)
    grid_expected2 = ImageGrid(expected2, rows=5, cols=10)




    #get a window
    window = grid.get(row=0, col=0)
    window_expected = grid_expected.get(row=0, col=0)
    window_fail = grid_fail.get(row=0, col=0)
    window_expected2 = grid_expected2.get(row=0, col=0)


    #get the statistics
    #convert the stats to a pandas dataframe
    observed_data = TextureEnergy.create_stats_dataframe(window, blur=True)
    print('\n\n----Observed----')
    print(observed_data)


    #now create the dataframe for the expected
    expected_data = TextureEnergy.create_stats_dataframe(window_expected, blur=True)
    print('\n\n----Expected----')
    print(expected_data)


    #dataframe for the
    fail_data = TextureEnergy.create_stats_dataframe(window_fail, blur=True)
    print("\n\n----Failure Window----")
    print(fail_data)


    expected2_data = TextureEnergy.create_stats_dataframe(window_expected2, blur=True)
    print("\n\n----Better Observed Image----")
    print(expected2_data)


    print("\n\n----Chisquare for both unobstructed observed images----")
    cols =['mean', 'std', 'skewness', 'kurtosis', 'entropy']
    d1 = observed_data[cols].to_numpy()
    d2 = expected2_data[cols].to_numpy()
    chi = stats.chisquare(DataProcessing.convert_to_percent(d1, 0).ravel(), DataProcessing.convert_to_percent(d2, 0).ravel())
    print(chi)





def office():
    # read in and align images
    image_expected = cv2.imread("images/rawdisplay.png", cv2.IMREAD_GRAYSCALE)
    image_observed = cv2.imread("images/picture-of-display.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("before", image_observed)
    # cv2.imshow("after", image_observed)
    image_expected, image_observed = DataProcessing.align_images(image_expected, image_observed)
    _, fail = DataProcessing.align_images(image_expected, cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE))
    _, expected2 = DataProcessing.align_images(image_expected, cv2.imread('images/picture1.png', cv2.IMREAD_GRAYSCALE))
    expected2 = cv2.medianBlur(expected2, 5)




    #image_expected is unobstructed screenshot

    #fail is obstructed picture

    grid_expected = ImageGrid(image_expected, rows=5, cols=10)
    grid_observed = ImageGrid(fail, rows=5, cols=10)


    cv2.imshow('expected', grid_expected.mosaic())
    cv2.imshow('observerd', grid_observed.mosaic())


    cv2.waitKey()










if __name__ == '__main__':
    """entry point"""
    #test_visualization()
    #grab_macro_of_sample()
    #grab_all_macro_stats()
    #test_mosaic()
    #first_chi_squared()
    grab_stats_and_visualize()
    #office()




#take uncompressed image

#look into the different edge types for poencvs filter2d function wwith the convolution
#Convolutiona nd then split vs other way around.