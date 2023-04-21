# VIP_Project

# evaluate.py

To change the module folder based on the member's pipeline, please change the module based on the members name as in:

![image](https://user-images.githubusercontent.com/70759210/216228256-5fe61c66-1cf9-4a4c-acfc-4f347f8a32e5.png)

1. Boe Chang Horn, 1181103320 : boe.main


Make sure that the datasets folder path is just outside the project folder.

![image](https://user-images.githubusercontent.com/70759210/216230017-f6c75a26-4666-463c-ac3b-99ad949cc543.png)

Then, to change the dataset for running the evaluation code, please target the specfic dataset name with:

![image](https://user-images.githubusercontent.com/70759210/216228742-c7b59955-53c2-4dcf-9760-ad667b55571c.png)

1. dataset, numImages = 400
2. blur_images, numImages = 33
3. bright_light_images, numImages = 340
4. dim_light_images, numImages = 60
5. isolate_detection_error_images, numImages = 60

After confirming the module folder and target dataset, please run "python evaluate.py"

# evaluateROI.py

Similar steps required for changing the member's pipeline:

![image](https://user-images.githubusercontent.com/70759210/216230180-78dc02a2-ddf2-4794-9655-f0b805485f73.png)

And changing the dataset folder path:

![image](https://user-images.githubusercontent.com/70759210/216230237-046a6540-f76e-49ef-82f8-71745ffa17c3.png)

After confirming the module folder and target dataset, please run "python evaluateROI.py"

# boe/main.py

Please make sure to install the libraries based on the "requirements.txt" file under the folder "boe/"

To choose which pipeline to run, kindly change the following code to either:
![image](https://user-images.githubusercontent.com/70759210/215977736-d746cfa4-e61d-44d4-b7e1-620927e2b55a.png)

1. WPOD-net + MobileNets : pipeline = 1
2. WPOD-net + Pytesseract : pipeline = 2

To ensure pytesseract runs in _windows_, please do the following:

a) Install the "tesseract-ocr-w64-setup-5.3.0.20221222.exe" program

b) Based on the installation path, locate the file "tesseract.exe", and update the filepath under boe/main.py

![image](https://user-images.githubusercontent.com/70759210/215977399-9e012bfe-efba-4c07-a53a-ec0e1be24236.png)

For my example, it is installed under C:\Program Files\Tesseract-OCR\tesseract.exe


