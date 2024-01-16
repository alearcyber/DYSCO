"""
This file just has the ConfusionMatrix object.
It is more or less a data structure for the information that would be represented by a confusion matrix.
It is designed to be instantiated such that the data can be accumulated within the object as a test is being
performed. It also provides convenient functions for calculating accuracy, sensitivity, and specificty.
"""

class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0


    def register(self, predicted, actual):
        """
        This function is used for adding data to the confusion matrix. The idea is that one can iterate over their
        results, and use this function to accumulate each individual result in an instance of the ConfusionMatrix
        object. The object will then be used later to analyze the aggregate information.
        :param predicted: bool, truth of the predicted value.
        :param actual: bool, truth of the actual value, the ground truth.
        """
        if predicted and actual:
            self.tp += 1
        elif predicted and (not actual):
            self.fp += 1
        elif (not predicted) and actual:
            self.fn += 1
        else:
            self.tn += 1


    def accuracy(self):
        """
        :return: Returns the accuracy of the ConfusionMatrix.
        """
        total = self.tp + self.tn + self.fn + self.fp
        assert total > 0, "Error, you have no data in your confusion matrix."
        return (self.tp + self.tn) / total

    def sensitivity(self):
        """
        :return: Returns the sensitivity of the ConfusionMatrix. Sensitivity is the ability to discern positive
        outcomes.
        """
        return self.tp / (self.tp + self.fn)
        pass

    def specificity(self):
        """
        :return: Returns the specificity of the ConfusionMatrix. Specificity is the ability to discern negative
        outcomes.
        """
        return self.tn / (self.tn + self.fp)


    def __str__(self):
        """
        String representation of the confusion matrix
        """
        out = f"---ConfusionMatrix---\n" \
              f"Accuracy: {self.accuracy()}\n" \
              f"Sensitivity: {self.sensitivity()}\n" \
              f"Specificity: {self.specificity()}\n" \
              f"tp: {self.tp} | tn: {self.tn} | fp: {self.fp} | fn: {self.fn}"
        return out




def test1():
    """
    This is a simple test for the confusion matrix object
    """
    c = ConfusionMatrix()
    c.register(True, True)
    c.register(True, True)
    c.register(True, False)
    c.register(True, False)
    c.register(False, True)
    c.register(False, True)
    c.register(False, False)
    c.register(False, False)
    print(c)






if __name__ == "__main__":
    test1()


