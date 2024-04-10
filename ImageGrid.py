

class ImageGrid:
    def __init__(self):
        self.rows = 0
        self.cols = 0
        self.mode = None  # 'col' or 'row'

        self.images = []
        """ This is the way images are going to be stored
        [
            [i1, i2, i3],
            [i4, i5, i6],
            [i7, i8, i9],
        ]
        """


    def get_image(self, row, col):
        pass

    def insert_column(self, images):
        """
        Inserts a column of images
        :param images:
        """
        if self.mode is None:
            self.mode = 'col'
            self.rows = len(images)
            self.images = [[image] for image in images]

        elif self.mode == 'row':
            assert False, "Error, ImageGrid already in column mode."

        else:
            assert len(images) == self.rows, "Error, inserting column with unexpected height."
            for row in self.images:
                row.append(images)

        self.cols += 1





    def show(self):
        """
        Display currently held images
        """
        pass

