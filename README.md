# YOLOv8-Object-Detection

## Part I: Introduction
This project implements a Python program for car recognition in an image using the YOLOv8 algorithm.

## Part II: Program Installation

To install and use the YOLOv8 algorithm, follow these instructions:

1. **Download YOLOv8 Source Code from GitHub**: To use YOLOv8, we need to download the source code from the YOLOv8 GitHub repository. The YOLOv8 source code is publicly available on GitHub. Follow these steps:
   - Step 1: Access the YOLOv8 GitHub repository [here](https://github.com/ultralytics/ultralytics).
  

4. **Prepare the Data**: To train YOLOv8 on any dataset, you need two main components:
   - Data directory: Prepare a directory that contains the dataset. Building a custom dataset can be a painful process. It might take dozens or even hundreds of hours to collect images, label them, and export them in the proper format. Fortunately, Roboflow makes this process straightforward. If you only have images, you can label them in Roboflow. For details please refer 
   
   \href{https://fulldataalchemist.medium.com/building-your-own-real-time-object-detection-app-roboflow-yolov8-and-streamlit-part-1-f577cf0aa6e5}{here}

  
     ```

   - The .yaml file: Prepare a .yaml file that contains information about the dataset mentioned above. You can refer to some example .yaml files provided by the YOLOv8 author:

     ![Important fields in the .yaml file](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.020.jpeg)

     **Figure 17:** Important fields in the .yaml file

     For the human dataset, you need to create a new .yaml file (e.g., data.yaml) and fill in the corresponding information (the provided dataset folder structure). You can do this manually or use the following code to create it automatically. Note that the .yaml file should be placed inside the human_detection_dataset folder:

     ```python
     import yaml

     dataset_info = {
       'train': './train/images',
       'val': './val/images',
       'nc': 1,
       'names': ['Human']
     }

     with open('./human_detection_dataset/data.yaml', 'w+') as f:
       doc = yaml.dump(dataset_info, f, default_flow_style=None, sort_keys=False)
     ```

     Each dataset may have different information, so you need to adjust it according to your dataset (number of classes, class names, data directory paths, etc.).

5. **Perform Training**: After completing the preparation steps, you can start the training process with the prepared dataset. Execute the following command (replace the data.yaml file name if using a different dataset):

   ```python
   !yolo train model=yolov8s.pt data=./human_detection_dataset/data.yaml epochs=20 imgsz=640
   ```
   After completing the training process, you will see the output displayed on the screen as shown in the following illustration:

   ![Output after training](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.023.png)

   **Figure 18:** Output displayed after completing the training process

   Check the ./runs directory, and you will find a ./detect/train file. This file contains the output of YOLOv8.

   ![train directory](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.024.png)

   **Figure 19:** The train directory
6. **Performing Detection (Prediction) with Trained Model**

   To use the trained model on any arbitrary image, use the following command:

   ```python
   # With uploaded image
   !yolo predict model=<weight_path> source=<image_path>
   ```
   Where:
   - `<weight_path>`: Path to the weight file of the trained model. You can find it in the output displayed after the YOLOv8 training process (refer to Figure 20 below).

   ![Weight file path](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.026.png)

   Figure 20: Path to the weight file of the trained model. Note: Choose the best.pt file.

   - `<image_path>`: Path to the input image file.

   The following is an illustrated result after executing the above code:

   ![Detection result](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.027.png)

   In addition to the image source, you can also input different data types as represented by the different parameters:

   ![Input data types](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.028.png)

   Figure 21: Overview of different input data types supported by YOLOv8 for prediction
   
   Predict on sample.jpg => sameple_predict.jpg
   ![sample](./sample.jpg) ![sample_predict](./sample_predict.jpg)
   
   Predict on URL [link](https://www.youtube.com/watch?v=MsXdUtlDVhk)
   Result: https://youtu.be/B8HJfROv_jM

   This step concludes the tutorial on using YOLOv8. For the project's objective, you only need to apply the provided instructions for the human data (with    some necessary modifications) and proceed to step 7.
7. **OPTIONAL**: This section will discuss some additional aspects of YOLOv8, including parameters in the training command, model evaluation, and data labeling.

- **Parameters in the training command**: The training command in step 6 has default parameters, which you can customize according to your preferences. Different parameter values will yield different model performances. Here are the basic meanings of some parameters:

	- `img`: The size of the training image. The training and testing images will be resized to the specified size, which is set to 640 by default. You can experiment with different image sizes.

	- `batch`: During the training process, models can either read the entire training data at once or read it in batches. The default value is 64, which means the training dataset will be divided into batches of 64 samples. You can set different values as 2n (n ≥ 0).

	- `epochs`: The number of times the training process iterates over the dataset.

	- `data`: Information about the training dataset in a .yaml file.

	- `weights`: The pretrained model file to be used. You can download and use different pretrained model files from this [list](https://docs.ultralytics.com/tasks/detect/#models).

- **Model evaluation**: As mentioned before, the performance of the model can vary with different parameter values. To quantitatively evaluate the models and find the best-performing one, you can execute the following command:

![Model evaluation](./image/Aspose.Words.6e6b9928-8ec9-4154-9519-82ec9d040593.029.png)

Figure 22: Evaluation results on the validation set


