import os
import urllib.request
import gzip
import shutil

def download_file(url: str, filename: str):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download face detection model files
    deploy_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    download_file(deploy_url, 'models/deploy.prototxt')
    download_file(weights_url, 'models/res10_300x300_ssd_iter_140000.caffemodel')
    
    # Download face recognition model
    face_model_url = "http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.t7"
    download_file(face_model_url, 'models/nn4.small2.v1.t7')
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main() 