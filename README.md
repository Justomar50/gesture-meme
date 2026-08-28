# MEME MATCH

Real-time Hand & Face Gesture Recognition using OpenCV and MediaPipe.

## Built With
- Python 3
- OpenCV
- MediaPipe (Hands & Face Mesh)

## How It Works
- OpenCV opens your webcam feed frame-by-frame.
- MediaPipe detects 21 hand landmarks and facial points.
- The app compares landmark coordinates (e.g., finger height or distance to the nose) to detect gestures.
- A corresponding meme/image is displayed side-by-side with your webcam feed.

## Setup & How to Run

1. **Install required libraries:**
   Open your terminal/command prompt and run:
   ```bash
   pip install -r requirements.txt
   ```
2. **to run the code**
 
    ```bash
    python meme_match.py
    ```
    Press `q` or `Esc` to quit.
  ----  
  # Images
  Sample meme images are included in the `images/` folder.
*Feel free to swap them with your own!*
  