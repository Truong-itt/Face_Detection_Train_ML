# Face Detection From Video - Machine Learning Training Model

This project demonstrates the use of machine learning techniques, specifically convolutional neural networks (CNNs), to detect human faces in video streams using Python, OpenCV, and TensorFlow. The system is trained on a custom dataset that is suitable for further development for various face recognition applications.

## Repository Structure

- **haarcascades/**: Contains Haar Cascade XML files for face detection.
- **con2d_update.py**: Custom implementation of a convolutional neural network to understand CNN operations.
- **crossetropy.py**: Explains and demonstrates the calculation of cross-entropy loss.
- **image_example.jpg**: A sample image for testing and demonstration.
- **matrix_3d.py**: Includes functions for manipulating 3D matrices, useful for image data.
- **model_building.py**: The main script where the CNN model is built, trained, and saved.
- **requirements.txt**: Lists all necessary Python libraries for the project.
- **test_model.py**: Script to deploy the trained model on real-time video streams.

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-github-username>/Face_Detection_Train_ML.git
   cd Face_Detection_Train_ML
2. **Set up a Python environment:**
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix or MacOS
3. **Install dependencies:**
   pip install -r requirements.txt
4. **Download the dataset**
   https://drive.google.com/drive/folders/1Vs16_E1YhjQFdVYMeV8dTDARkt05ghUQ?usp=sharing
#Usage
1. **Train the model:**
   python model_building.py
2. **Test the model on real-time video:**
   python test_model.py




