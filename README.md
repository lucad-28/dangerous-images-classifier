# CNNs to classify potential dangerous objects in images

This project try to implement a CNN in R to classify images between dangerous and non-dangerous objects. The dataset is composed by 2 classes: pistolas and cuchillos. The dataset for train the model is composed by 100 images of each class. The images are 640x640 pixels, and it's part of the dataset ![armas Computer Vision Project](https://universe.roboflow.com/espe-wc2wk/armas-dixit).

Setup the environment:

1. Install Python 3.10.\*

2. Create a virtual environment:

```bash
C:\Users\Yo\AppData\Local\Programs\Python\Python310\python.exe -m venv cnns02env
```

3. Install necessary libraries:

```bash
pip install -r requirements.txt
```

4. Edit environment path in `train_script.R` file:

```python
# Replace with your virtual environment path
use_virtualenv("D:/Cursos/2024-II/Big Data - Tareas/Evidencia 1/sqlanaenv")
```
