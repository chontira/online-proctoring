import cv2
import numpy as np
import dlib
import face_recognition
import math
from datetime import datetime
import os

def average(li):
    return sum(li)/len(li)

def encode_test_taker_img(img_path):
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    face_encoding = face_recognition.face_encodings(image,face_locations)[0]
    return face_encoding

def prepare_video_for_face_reg(bgr_frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    bgr_small_frame = cv2.resize(bgr_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    return bgr_small_frame[:, :, ::-1]

def face_reg(rgb_small_frame):
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face = []
    for i,face_encoding in enumerate(face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face.append([face_locations[i],name])
    return face

def draw_face_reg(faces):
    for (top, right, bottom, left), name in faces:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == 'Unknown':
            color = (0, 0, 255) # red
        else:
            color = (255,0,0) #blue
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame

def get_14_face_landmark(face_loc):
    # for face_loc,name in faces:
        # if name != 'Unknown':
    top, right, bottom, left = face_loc
    top *= 4 
    right *= 4
    bottom *= 4 
    left *= 4
    face_location = (top-50, right+50, bottom+50, left-50)
    face_location = (top, right, bottom, left)
    # face_img = img[top-50:bottom+50, left-50:right+50]
    face_landmarks = face_recognition.face_landmarks(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),[face_location])
    # face_landmarks = face_recognition.face_landmarks(cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB))
    _14_points = np.array([
        face_landmarks[0]['left_eyebrow'][0],     # 17 left eyebrow left corner
        face_landmarks[0]['left_eyebrow'][4],     # 21 left eyebrow right corner
        face_landmarks[0]['right_eyebrow'][0],    # 22 right eyebrow left corner
        face_landmarks[0]['right_eyebrow'][4],    # 26 right eyebrow right corner
        face_landmarks[0]['left_eye'][0],         # 36 Left eye left corner
        face_landmarks[0]['left_eye'][3],         # 39 Left eye right corner
        face_landmarks[0]['right_eye'][0],        # 42 Right eye left corner
        face_landmarks[0]['right_eye'][3],        # 45 Right eye right corner
        face_landmarks[0]['nose_tip'][0],         # 31 Nose tip left
        face_landmarks[0]['nose_tip'][4],         # 35 Nose tip right
        face_landmarks[0]['top_lip'][0],          # 48 Left Mouth corner
        face_landmarks[0]['top_lip'][6],          # 54 Right mouth corner
        face_landmarks[0]['bottom_lip'][3],       # 57 center bottom lip 
        face_landmarks[0]['chin'][8]              # 8 Chin
    ], dtype="double")
    return _14_points

def head_pose(image_points):
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    angles[0, 0] = angles[0, 0] * -1
    return round(angles[2, 0],2), round(angles[0, 0],2), round(angles[1, 0],2)  # roll, pitch, yaw

def draw_head_pose(roll, pitch, yaw, head):
    text = 'roll: '+str(roll)+' pitch: '+str(pitch)+' yaw:'+str(yaw)
    cv2.putText(frame, text, (75,75), font, 2, (128, 255, 255), 3) # draw yellow number (up,down)
    cv2.putText(frame, head, (75,110), font, 2, (255, 255, 128), 3) # draw blue number (left,right)

disappear = False
stranger = False
head_leftright = False
head_down = False
object_detect = [False,False,False]
def alert(disappear_frame,stranger_frame, avg_pitch, avg_yaw, object_frame):
    global disappear
    global stranger
    global head_leftright
    global head_down

    # alert test_taker disappear
    if (disappear_frame == 10) & (disappear == False):
        disappear = True
        print(datetime.now().strftime("%H:%M:%S")+' disappear')
    elif (disappear_frame == 0) & (disappear == True):
        disappear = False
        print(datetime.now().strftime("%H:%M:%S")+' appear')

    # alert found stranger
    if (stranger_frame == 10) & (stranger==False):
        stranger = True
        print(datetime.now().strftime("%H:%M:%S")+' stranger appear')
    elif (stranger_frame == 0) & (stranger == True):
        stranger = False
        print(datetime.now().strftime("%H:%M:%S")+' stranger disappear')
    
    # alert head down
    if (avg_pitch < -8) & (head_down == False):
        head_down = True
        print(datetime.now().strftime("%H:%M:%S")+' head down')
    elif (avg_pitch > -8) & (head_down == True):
        head_down = False
        print(datetime.now().strftime("%H:%M:%S")+' head stright')
    
    # alert head turn left or right
    if (avg_yaw < -25) & (head_leftright == False):
        head_leftright = True
        print(datetime.now().strftime("%H:%M:%S")+' head left')
    elif (avg_yaw > 25) & (head_leftright == False):
        head_leftright = True
        print(datetime.now().strftime("%H:%M:%S")+' head right')
    elif (avg_yaw > -25) & (avg_yaw < 25) & (head_leftright == True):
        head_leftright = False
        print(datetime.now().strftime("%H:%M:%S")+' head stright')
    
    # alert found object
    for i in range(3):
        if (object_frame[i] == 10) & (object_detect[i]==False):
            object_detect[i] = True
            print(datetime.now().strftime("%H:%M:%S")+' found '+LABELS[i])
        elif (object_frame[i] == 0) & (object_detect[i] == True):
            object_detect[i] = False
            print(datetime.now().strftime("%H:%M:%S")+' not found '+LABELS[i])

def get_test_taker_img(id):
    for fname in os.listdir('test_taker_img'): 
        if id in fname:
            return fname

def load_object_detection():
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["model", "obj.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["model", "yolov4-tiny-custom_best.weights"])
    configPath = os.path.sep.join(["model", "yolov4-tiny-custom.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return LABELS,COLORS,net

def object_detection(frame):
    # determine only the *output* layer names that we need from YOLO
    # global net
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thres)
    boxes_new = []
    confidences_new = []
    classIDs_new = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            boxes_new.append(boxes[i])
            confidences_new.append(confidences[i])
            classIDs_new.append(classIDs[i])
    return boxes_new, confidences_new, classIDs_new

def draw_object_detection(boxes, confidences, classIDs, LABELS):
    for i in range(len(boxes)):
        if classIDs[i] != 3:
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

# Initialize some variables
process_this_frame = True
disappear_frame = 0
stranger_cnt = 0
stranger_frame = 0
conf = 0.5
thres = 0.3
yaw_list = []
pitch_list = []
object_frame =[0,0,0]

# load test taker img image and encode
stu_id = input('Enter your student ID: ')
test_taker_fname = get_test_taker_img(stu_id)
test_taker_face_encoding = encode_test_taker_img('test_taker_img/'+test_taker_fname)
known_face_encodings = [test_taker_face_encoding]
known_face_names = [test_taker_fname.split('.')[0]]

# load YOLO from disk
LABELS,COLORS,net = load_object_detection()

# 3D model points.
model_points = np.loadtxt("model/model_points.txt")

# start webcam
cap = cv2.VideoCapture(0)

# Initialize head pose variables
ret, img = cap.read()
# size = img.shape
size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
(H, W) = size
font = cv2.FONT_HERSHEY_SIMPLEX

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    bgr_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = bgr_small_frame[:, :, ::-1]
    # rgb_small_frame = prepare_video_for_face_reg(frame)
    
    # if process_this_frame:
    disappear_frame +=1
    object_frame[0] +=1
    object_frame[1] +=1
    object_frame[2] +=1

    faces = face_reg(rgb_small_frame)
    boxes, scores, classes = object_detection(bgr_small_frame)
    for c in np.arange(3):
        if (not (c in classes)):
            object_frame[c] = 0
    stranger_cnt = len(faces)
    for face_loc,name in faces:
        if name != 'Unknown':
            image_points = get_14_face_landmark(face_loc)
            roll, pitch, yaw = head_pose(image_points)
            yaw_list.append(yaw)
            pitch_list.append(pitch)
            if len(yaw_list) > 10:
                del yaw_list[0]
                del pitch_list[0]
            disappear_frame = 0
            stranger_cnt -= 1
    if stranger_cnt > 0:
        stranger_frame +=1
    else:
        stranger_frame = 0
    # process_this_frame = not process_this_frame
    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    head = ''
    # if yaw > 3:
    #     head=head+'right '
    # elif yaw < -3: 
    #     head=head+'left '
    # if pitch > 10:
    #     head=head+'up'
    # elif pitch < -8:
    #     head=head+'down'
    avg_pitch = round(average(pitch_list),2)
    avg_yaw = round(average(yaw_list),2)
    alert(disappear_frame,stranger_frame, avg_pitch, avg_yaw, object_frame)
    # print(avg_pitch, pitch_list)
    # print(avg_yaw, yaw_list)
    draw_object_detection(boxes, scores, classes, LABELS)
    draw_head_pose(roll, pitch, yaw, head)
    draw_face_reg(faces)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()