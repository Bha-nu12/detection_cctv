import cv2

def test_camera(index):
    print(f"\nTesting camera index {index}")
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_MSMF, "Media Foundation")
    ]
    
    for backend, name in backends:
        print(f"Trying {name} backend...")
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {index} works with {name} backend")
                cap.release()
                return True
            else:
                print(f"✗ Camera {index} opened but couldn't read frame with {name}")
            cap.release()
        else:
            print(f"✗ Camera {index} failed to open with {name}")
    return False

def main():
    print("Testing available cameras...")
    # Test first 5 indices
    for i in range(5):
        test_camera(i)

if __name__ == "__main__":
    main() 