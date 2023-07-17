"""
Contains the functionality for separating out an image into a grid.
Implemented as an object
"""
import cv2
import TextureEnergy
import numpy as np

class ImageGrid:
    def __init__(self, image, rows, cols):
        self.image = image
        self.rows = rows
        self.cols = cols
        self.subimages = []
        self._grid_out_image()
        self.dtype = image.dtype

    # split up an image into n x m,  n is across, m is up and down
    def _grid_out_image(self):
        """split the image into n x n sub-images, return in list, ordered like reading order"""
        height, width = self.image.shape[0], self.image.shape[1]  # extract width and height of the image
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
        img = self.image[y1:y2, x1:x2]
        return img


    #retreives an individual subimage
    def get(self, row, col):
        return self.subimages[row][col]



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
        for row in self.rows:
            stat_row = []
            for col in self.cols:
                stats = self.macro_statistics(row, col)
                stat_row.append(stats)
            one_up.append(stat_row)
        return np.array(one_up, dtype=np.uint8)









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
    cv2.waitKey()







if __name__ == '__main__':
    """entry point"""
    #test_visualization()
    #grab_macro_of_sample()
    test_mosaic()