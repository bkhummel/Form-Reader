# Form-Reader
Project to read handwritten text for a standardized form provided by Indiana Limestone

Dependencies:
  opencv-contrib-python
  Keras + Tensorflow
  Numpy

To run the form reader, clone this directory and run Main.py.


Here is the form we want to read
<p float="left">
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Scanned_Forms/Form_A.png" width = "500"/>
</p>
We use template matching to find the fields and their associated boxes.
To do this, I cropped the titles of each field and used them as templates.
<p float="left">
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Channel.jpg" width = "200"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Qdate.jpg" width = "200"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Cut.jpg" width = "200"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Length.jpg" width = "200"/>
</p>



Run template matching to get the text boxes to the right.
Example of template matching for 'Channel'
<p float="left">
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/template_matches/0.jpg" width = "500"/>
</p>



Now we have all boxes pertaining to the 'Channel' field.
<p float="left">
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Example_pics/Screenshot%20(19).png"/>
</p>



Run cv2.findContours on each box to get individual letters.
<p float="left">
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Example_pics/Screenshot%20(20).png"/>
</p>



Finally, feed each letter to the model and save predictions in specified text file ("predictions.txt").
This process repeats for all fields.


Unfortunately this did not turn out to be a good solution. While the form is fairly easy to process, the
model's predictions are far too unreliable to be used as an automatic process. Average accuracy is 80%.
