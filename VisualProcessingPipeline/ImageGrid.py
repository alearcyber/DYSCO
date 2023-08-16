"""
Contains the functionality for separating out an image into a grid.
Implemented as an object
"""
import cv2
import numpy as np

class ImageGrid:
    def __init__(self, image, rows, cols):
        self.image = image
        self.rows = rows
        self.cols = cols
        if type(image) is list:
            self.subimages = image[:]
            self.subwidth = self.subimages[0][0].shape[1]
            self.subheight = self.subimages[0][0].shape[0]
            self.dtype = image[0][0].dtype
        else:
            self.subimages = []
            self._grid_out_image(image)
            self.dtype = image.dtype
            self.subwidth = None
            self.subheight = None

    # split up an image into n x m,  n is across, m is up and down
    def _grid_out_image(self, image):
        """split the image into n x n sub-images, return in list, ordered like reading order"""
        height, width = image.shape[0], image.shape[1]  # extract width and height of the image
        m, n = self.rows, self.cols
        self.subheight = height // m
        self.subwidth = width // n

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
            x_crossections.append(self.subwidth * i)

        for i in range(m):
            y_crossections.append(self.subheight * i)

        for j in range(len(y_crossections)):
            y = y_crossections[j]
            for i in range(len(x_crossections)):
                x = x_crossections[i]
                x1, y1, x2, y2 = x, y, x + self.subwidth, y + self.subheight

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

    #set a different image for one of the subimages
    def set(self, image, row, col):
        #check for different size
        shape = self.get_subimage_shape()
        if image.shape != shape:
            image = cv2.resize(image, (shape[1], shape[0]), cv2.INTER_CUBIC)
        self.subimages[row][col] = image


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
