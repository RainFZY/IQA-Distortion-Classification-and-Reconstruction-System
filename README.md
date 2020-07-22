## IQA-and-Distortion-Classification-System
### IQA (Image Quality Assessment)

Use 5 models -- CNN, LeNet-5, ResNet18, ResNet34, VGG19 to respectively assess the input image's quality and get 5 scores. The final IQA score are the average of these 5 scores.

Models are provided in the folder `app/assess/models`

To check the training process or get your own model, follow the link: 

https://github.com/RainFZY/Image-Quality-Assessment-By-Multiple-Models

### Image Distortion Classification

Classify the distortion of input image into 3 classes: 

**noise** (wn), **blur** (gblur), **JPEG compression** (jpeg)

Provide each classification's confidence coefficient.

### Start the Django Service

1. Enter the project folder in cmd

   ```
   cd ../IQA-and-Distortion-Classification-System
   ```

2. run

   ```
   python manage.py runserver
   ```

3. Enter http://127.0.0.1:8000/ in your browser

### Test Demo

***Home page***

![](https://github.com/RainFZY/IQA-and-Distortions-Processing-System/blob/master/images/screen_shot/example1.png)

***Test a blurred image***

![](https://github.com/RainFZY/IQA-and-Distortions-Processing-System/blob/master/images/screen_shot/example2.png)

The IQA scores and the distortion classification results are listed in the right area

***Test a JPEG compression image***

![](https://github.com/RainFZY/IQA-and-Distortions-Processing-System/blob/master/images/screen_shot/example3.png)

***Test a noisy image***

![](https://github.com/RainFZY/IQA-and-Distortions-Processing-System/blob/master/images/screen_shot/example4.png)