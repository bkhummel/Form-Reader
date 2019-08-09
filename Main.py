import cv2
import numpy as np
import os
import CNN


class FormReader:

    def __init__(self, form, file):
        self.form = cv2.imread(form);
        self.file = open(file, "w")
        self.model = CNN.CNN('1').load_model()

    def readForm(self):
        """Iterate through all templates"""
        for t in os.listdir('Templates/'):
            template = cv2.imread('Templates/' + t, 0)
            formCopy = self.form.copy()

            self.get_boxes(template, formCopy)

            """Iterate through all boxes of each template"""
            for box in os.listdir('Boxes/'):
                box = cv2.imread('Boxes/' + box)
                self.getLetters(box)

                predictions = []
                """iterate through each letter in those boxes"""
                for letter in os.listdir('Letters/'):
                    letter = cv2.imread('Letters/' + letter)
                    letter = self.letterReshape(letter)
                    """Make Predictions for Each letter"""
                    p = self.makePrediction(letter)
                    predictions.append(p)
                self.update_fields(predictions, t)
                self.clearDir('Letters/')
            #clearDir('Boxes/')
        self.file.close()
        return 0


    """
    clears directory for next set of inputs
    """
    def clearDir(self, dir):
        for f in os.listdir(dir):
            os.remove(dir + f)
        return None


    """
    
    """
    def get_boxes(self, template, form):
        formCopy = form.copy()
        boxImg = self.drawBoxes(template, form)
        self.saveBoxes(boxImg, formCopy)
        return None


    """takes a template and form, runs template matching, and draws red rectangles over the boxes
    to the right of the matched templates."""
    def drawBoxes(self, template, form):
        grayForm = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(grayForm, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(form, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        return form


    """takes the form with red rectangles drawn on it, turns it into a binary image, takes the contours, and saves them to Boxes/ """
    def saveBoxes(self, img, form):
        img[np.where((img==[255,255,255]).all(axis=2))] = [0,0,0]
        img[np.where((img==[0,0,255]).all(axis=2))] = [255,255,255]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sorted_cnts = self.sortContours(gray, 1)

        idx = 0
        for cnt in sorted_cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            box = form[y+5:y+h-5, x+w+5:x+w+245]
            cv2.imwrite('Boxes/' + str(idx).zfill(4) + '.jpg', box)
            idx += 1
        return 0


    """ takes raw image of letter, binarizes it, and reshapes it to (1,28,28,1). Returns binary image """
    def letterReshape(self, letter):
        """padding if smaller"""
        s = letter.shape
        if s[0] < 28:
            top = 28 - s[0]
        else:
            top = 0
        if s[1] < 28:
            left = 28 - s[1]
        else:
            left = 0
        # if image is smaller than 28x28, use copyMakeBorder to fill. This avoids warping the image beyond recognition
        letter = cv2.copyMakeBorder(letter, top, 0, left, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]);
        img = cv2.resize(letter, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        bin1 = self.binarize(img, 150)
        bin1 = np.reshape(bin1, (1, 28, 28, 1))

        return bin1


    """
    Saves individual handwritten letters from boxes
    input: box = field entry box with handwriting
    returns: None
    """
    def getLetters(self, box):
        idx = 0
        bin1 = self.binarize(box, 150)
        binDilated = cv2.dilate(bin1,(3,3),iterations = 8)
        sorted_cnts = self.sortContours(binDilated, 0)

        for cnt in sorted_cnts:
            x,y,w,z = cv2.boundingRect(cnt)
            letter = box[y:y+z+5, x:x+w+5]
            a,_,_ = letter.shape
            if a > 25:
                cv2.imwrite('Letters/' + str(idx).zfill(4) + '.jpg', letter)
                idx += 1

        return None


    """
    input: img = image, d = axis at which you want to sort the contours.
    Returns: a list of sorted contours
    """
    def sortContours(self, img, d):
        contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_cnts = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[d])
        return sorted_cnts

    """
    Turns input image into binary image.
    input: img = image to binarize, thresh = threshold for binarizing
    returns: None
    """
    def binarize(self, img, thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, bin1 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        return bin1

    """
    input: model = CNN model, letters = images of individual letters
    returns: Keras.model prediction
    """
    def makePrediction(self, letter):
        p = self.model.predict_classes(letter)
        return p

    """
    updates textfile with appropriate prediction along with its associated field.
    """
    def update_fields(self, predictions, boxname):
        file = self.getfile()
        file.write(boxname + ": ")

        for p in predictions:
            file.write(str(p) + ', ')
        file.write('\n')
        return None

    def getfile(self):
        return self.file

    def getform(self):
        return self.form

a = FormReader('Scanned_Forms/Form_A.png',"predictions.txt")
a.readForm()