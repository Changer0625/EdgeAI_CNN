import cv2
import numpy as np
from typing import Iterator, Tuple, Optional

def load_video(path:str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture('original_video/full.mp4')
    return cap

def load_model(prototype_path:str, model_path:str) -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromCaffe(prototype_path, model_path)
    if net.empty():
        print('Failed model load')
    else:
        print('Success model load')
    return net

def create_video_writer(output_path:str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (int(video_capture.get(3)), int(video_capture.get(4))), True)
    return writer

def video_iterable(cap:cv2.VideoCapture) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield ret, frame

class coordinate_rectangle():
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
    
    def contains(self, other:"coordinate_rectangle") -> bool:
        return (self.start_x <= other.start_x and self.end_x >= other.end_x) \
           and (self.start_y <= other.start_y and self.end_y >= other.end_y)
    
    def offset(self, x_offset:int, y_offset:int) -> None:
        self.start_x += x_offset
        self.end_x += x_offset
        self.start_y += y_offset
        self.end_y += y_offset

def put_rectangle(frame:np.ndarray, rectangle:coordinate_rectangle, red:int, green:int, blue:int) -> None:
    cv2.rectangle(frame, (rectangle.start_x, rectangle.start_y), (rectangle.end_x, rectangle.end_y), (blue, green, red), 2)

def door_status_check_open(door_frame:np.ndarray, door_color_threshold:int) -> bool:
    """
    Check elevator door open/close status,
    result is sensitive to threshold

    Paramaters:
    door_frame (np.ndarray): Image that contains only door

    Returns:
    True if door is open, otherwise False
    """
    door_mean_color = np.mean(door_roi, axis=(0,1))
    elevator_doors_open = door_mean_color[1] - door_mean_color[0] > door_color_threshold
    return elevator_doors_open

def count_people(frame:np.ndarray, model:cv2.dnn_Net, elevator_doors_open:bool) -> Tuple[int, int]:
    # Get input for model + run model
    roi = frame[roi_rect.start_y:roi_rect.end_y, roi_rect.start_x:roi_rect.end_x] # ROI frame
    blob = cv2.dnn.blobFromImage(roi, 0.007843, (300, 300), 127.5, swapRB=True) # network input
    model.setInput(blob) # set input
    detections = model.forward() # network output

    # Process output
    external_people, internal_people = 0, 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # confidence
        class_id = int(detections[0, 0, i, 1]) # object type
        if confidence > 0.4 and class_id == 15: # class id:15 -> person
            box = detections[0, 0, i, 3:7] * np.array([roi.shape[1], roi.shape[0], roi.shape[1], roi.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            person_rectangle = coordinate_rectangle(x1, y1, x2, y2)
            person_rectangle.offset(roi_rect.start_x, roi_rect.start_y) # Result rectangle

            if elevator_doors_open and elevator_rect.contains(person_rectangle): # People inside elevator detected
                internal_people += 1
                put_rectangle(frame, person_rectangle, 255, 150, 100) # Light Orange: Inside
            else: # People outside elevator detected
                external_people += 1
                put_rectangle(frame, person_rectangle, 100, 150, 255) # Light Blue: Outside
    return external_people, internal_people

#NOTE: ROI -> Region of Interest

#Paths
INPUT_VIDEO_PATH = 'original_video/full.mp4'
MODEL_PROTOTYPE_PATH = 'MobileNet-SSD/deploy.prototxt'
MODEL_PATH = 'MobileNet-SSD/mobilenet_iter_73000.caffemodel'
OUTPUT_VIDEO_PATH = 'output2_video.mp4'

def main():
    #I/O
    video_capture = load_video(INPUT_VIDEO_PATH)
    model = load_model(MODEL_PROTOTYPE_PATH,MODEL_PATH)
    output_video_writer = create_video_writer(OUTPUT_VIDEO_PATH)

    #Manual Parameters
    roi_rect = coordinate_rectangle(160, 80, 730, 500) #region of interest (ROI), coordinates of video where we want to process
    elevator_rect = coordinate_rectangle(190, 90, 360, 400) #region of interest, coordinates of video which contains elevator
    door_x, door_y, door_side_length = 260, 130, 5 #small square of elevator door for checking status
    door_color_threshold = 10 # elevator checked custom threshold

    #Main functionality
    max_total_people = 0
    max_internal_people = 0 #people count inside elevator, uses max because some people may be undetected after being detected previously
    for ret, frame in video_iterable(video_capture):
        # Region of interest
        put_rectangle(frame, roi_rect, 255, 255, 0) # Yellow: All ROI
        put_rectangle(frame, elevator_rect, 100, 100, 100) # Gray: Elevator ROI

        # Get door status
        door_roi = frame[door_y-door_side_length:door_y+door_side_length, door_x-door_side_length:door_x+door_side_length] # get elevator roi frame
        elevator_doors_open = door_status_check_open(door_roi, door_color_threshold) # get elevator door status
        
        # Detect people function
        external_people, internal_people = count_people(frame, model, elevator_doors_open) # Count external, internal people, also puts rectangle when detected

        # Process model result
        max_total_people = max(max_total_people, external_people+internal_people)
        if not elevator_doors_open:
            external_people = max_total_people
            internal_people = 0
        else:
            internal_people = max(internal_people, max_total_people-external_people)
            max_internal_people = max(max_internal_people, internal_people)
            external_people = max_total_people-max_internal_people # Hakuna matata

        # Final report
        text = f'Door Status: {"Open" if elevator_doors_open else "Close"}' 
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Write elevator status
        text = f'People outside Elevator: {external_people}'
        cv2.putText(frame, text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # Write external people count
        text = f'People in Elevator: {max_internal_people}'
        cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Write internal people count

        # Write output video
        output_video_writer.write(frame)

    #cleanup
    video_capture.release()
    output_video_writer.release()
    print("Program Ended")

if __name__ == '__main__':
    main()