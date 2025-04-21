import cv2
import numpy as np

cap = cv2.VideoCapture('original_video/full.mp4')

net = cv2.dnn.readNetFromCaffe('MobileNet-SSD/deploy.prototxt','MobileNet-SSD/mobilenet_iter_73000.caffemodel')

output_video = cv2.VideoWriter('output2_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (int(cap.get(3)), int(cap.get(4))), True)

if net.empty():
    print('No data')
else:
    print('success model load')


max_ip=0
roi_startX, roi_startY, roi_endX, roi_endY = 180, 100, 730, 400
ele_startX, ele_startY, ele_endX, ele_endY = 200, 100, 350, 400
door_x, door_y = 260,130
door_radius = 5
door_color_threshold = 10
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    door_roi = frame[door_y-door_radius:door_y+door_radius, door_x-door_radius:door_x+door_radius]
    door_mean_color = np.mean(door_roi, axis=(0,1))

    if door_mean_color[1] - door_mean_color[0] > door_color_threshold:
        elevator_doors_open = True
        roi_startX = 330
    else:
        elevator_doors_open = False
        roi_startX = 180

    door_status = 'Open' if elevator_doors_open else 'Close'
    cv2.putText(frame, f'Door Status: {door_status}', (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    roi = frame[roi_startY:roi_endY, roi_startX:roi_endX]
    ele = frame[ele_startY:ele_endY, ele_startX:ele_endX]
    blob = cv2.dnn.blobFromImage(roi, 0.007843, (300, 300), 127.5, swapRB=True)   
    ele_blob = cv2.dnn.blobFromImage(ele, 0.007843, (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    detections = net.forward()


    cv2.rectangle(frame, (180, roi_startY), (roi_endX, roi_endY), (0, 255, 255), 2)

    ep = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            class_id = int(detections[0, 0, i, 1])

            # 計算偵測框在 ROI 中的位置
            box = detections[0, 0, i, 3:7] * np.array([
                roi.shape[1], roi.shape[0], roi.shape[1], roi.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            # 加回偏移量到整張 frame 上
            x1 += roi_startX
            x2 += roi_startX
            y1 += roi_startY
            y2 += roi_startY

            if class_id == 15:  # class_id 15 是 person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                ep += 1

    text = f'Externel People: {ep}'
    cv2.putText(frame, text, (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if elevator_doors_open:
        net.setInput(ele_blob)
        detections = net.forward()
        ip = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                class_id = int(detections[0, 0, i, 1])

                # 計算偵測框在 ROI 中的位置
                box = detections[0, 0, i, 3:7] * np.array([
                    ele.shape[1], ele.shape[0], ele.shape[1], ele.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")

                # 加回偏移量到整張 frame 上
                x1 += ele_startX
                x2 += ele_startX
                y1 += ele_startY
                y2 += ele_startY

                if class_id == 15:  # class_id 15 是 person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    ip += 1
        if ip > max_ip:
            max_ip=ip
    text = f'People in Elevator: {max_ip}'
    cv2.putText(frame, text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_video.write(frame)

cap.release()
output_video.release()
print("Success video output")