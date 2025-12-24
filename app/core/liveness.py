import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

def check_liveness(video_path: str) -> Dict[str, Any]:
    """
    Analyzes a video file to determine if the subject is 'live' by detecting face movement.
    
    The algorithm tracks the center of the largest detected face across frames.
    It attempts to detect faces in the original orientation, and if that fails,
    it tries 90-degree and 270-degree rotations to handle mobile uploads.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - status (str): 'success', 'fail', or 'error'
            - is_alive (bool): True if movement exceeds threshold (only on success)
            - max_movement_pixels (float): Maximum distance moved between frames (only on success)
            - message (str): Human-readable result message
            - debug (str): Debugging information about frames processed
    """
    cap = cv2.VideoCapture(video_path)
    
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    previous_center: Optional[np.ndarray] = None
    movement_detected = False
    max_movement = 0.0
    frames_read = 0
    faces_found_count = 0
    
    MOVEMENT_THRESHOLD = 15 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Try detecting face normally
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # 2. If no face, try rotating 90 degrees (Fix for phone videos)
            if len(faces) == 0:
                gray_rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                faces = face_cascade.detectMultiScale(gray_rotated, 1.1, 5)
            
            # 3. If STILL no face, try rotating 270 degrees
            if len(faces) == 0:
                gray_rotated_2 = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                faces = face_cascade.detectMultiScale(gray_rotated_2, 1.1, 5)

            if len(faces) > 0:
                faces_found_count += 1
                # Take the largest face found
                (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                
                current_center = np.array([x + w//2, y + h//2])

                if previous_center is not None:
                    distance = np.linalg.norm(current_center - previous_center)
                    if distance > max_movement:
                        max_movement = float(distance)
                    
                    if distance > MOVEMENT_THRESHOLD:
                        movement_detected = True

                previous_center = current_center

    except Exception as e:
        logger.exception(f"Error processing video for liveness check: {str(e)}")
        return {"status": "error", "message": str(e)}
        
    finally:
        # Ensure resources are released even if an error occurs
        cap.release()
        
    # DEBUGGING: This message helps us know why it failed
    debug_msg = f"Processed {frames_read} frames. Found faces in {faces_found_count} frames."
    logger.info(debug_msg)

    if faces_found_count == 0:
        logger.warning(f"Liveness check failed: No faces found. {debug_msg}")
        return {
            "status": "fail",
            "message": "No face detected at all. Video might be too dark, too far, or heavily rotated.",
            "debug": debug_msg
        }

    result_message = "Liveness confirmed" if movement_detected else "Face detected, but no movement."
    logger.info(f"Liveness check complete: {result_message}. Max movement: {max_movement}")

    return {
        "status": "success",
        "is_alive": movement_detected,
        "max_movement_pixels": max_movement,
        "message": result_message,
        "debug": debug_msg
    }