# Form-Reader
Project to read handwritten text for a standardized form provided by Indiana Limestone


Here is the form we want to read
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Scanned_Forms/Form_A.png" width = "500"/>

We use template matching to find the fields and their associated boxes.
To do this, I cropped the titles of each field and used them as templates.
<p>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Channel.jpg" width = "300"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Qdate.jpg" width = "300"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Cut.jpg" width = "300"/>
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Templates/Lenth.jpg" width = "300"/>
</p>

Run template matching to get the text boxes to the right.
Example of template matching for 'Channel'
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/template_matches/0.jpg" width = "500"/>

Save these boxes to 'Boxes/' folder
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Example_pics/Screenshot%20(19).png"/>

Run cv2.findContours on each box to get individual letters.
<img src = "https://github.com/bkhummel/Form-Reader/blob/master/Example_pics/Screenshot%20(20).png"/>

Finally, feed each letter to the model and save predictions in specified text file ("predictions.txt").
