import cv2
import numpy as np
import dlib
import face_recognition
import math
from datetime import datetime
import os

def encode_examer_img(img_path):
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

# def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
#     """Draw a 3D box as annotation of pose"""
#     point_3d = []
#     dist_coeffs = np.zeros((4,1))
#     rear_size = 1
#     rear_depth = 0
#     point_3d.append((-rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, -rear_size, rear_depth))

#     front_size = img.shape[1]
#     front_depth = front_size*2
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d.append((-front_size, front_size, front_depth))
#     point_3d.append((front_size, front_size, front_depth))
#     point_3d.append((front_size, -front_size, front_depth))
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

#     # Map to 2d img points
#     (point_2d, _) = cv2.projectPoints(point_3d,
#                                       rotation_vector,
#                                       translation_vector,
#                                       camera_matrix,
#                                       dist_coeffs)
#     point_2d = np.int32(point_2d.reshape(-1, 2))
    

#     # # Draw all the lines
#     # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
#     k = (point_2d[5] + point_2d[8])//2
#     # cv2.line(img, tuple(point_2d[1]), tuple(
#     #     point_2d[6]), color, line_width, cv2.LINE_AA)
#     # cv2.line(img, tuple(point_2d[2]), tuple(
#     #     point_2d[7]), color, line_width, cv2.LINE_AA)
#     # cv2.line(img, tuple(point_2d[3]), tuple(
#     #     point_2d[8]), color, line_width, cv2.LINE_AA)
    
#     return(point_2d[2], k)

def head_pose(image_points):
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    angles[0, 0] = angles[0, 0] * -1
    return round(angles[1, 0],2), round(angles[0, 0],2), round(angles[2, 0],2)  # roll, pitch, yaw

def draw_head_pose(pitch, yaw, head):
    text = 'pitch: '+str(pitch)+' yaw:'+str(yaw)
    cv2.putText(frame, text, (75,75), font, 2, (128, 255, 255), 3) # draw yellow number (up,down)
    cv2.putText(frame, head, (75,110), font, 2, (255, 255, 128), 3) # draw blue number (left,right)

disappear = False
stranger = False
def alert(disappear_frame,stranger_frame):
    global disappear
    global stranger
    # alert examer disappear
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

def get_examer_img(id):
    for fname in os.listdir('examer'): 
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

# load examer image and encode
stu_id = input('Enter your student ID: ')
examer_fname = get_examer_img(stu_id)
examer_face_encoding = encode_examer_img('examer/'+examer_fname)
known_face_encodings = [examer_face_encoding]
known_face_names = [examer_fname.split('.')[0]]

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
    faces = face_reg(rgb_small_frame)
    boxes, scores, classes = object_detection(bgr_small_frame)
    stranger_cnt = len(faces)
    for face_loc,name in faces:
        if name != 'Unknown':
            image_points = get_14_face_landmark(face_loc)
            roll, pitch, yaw = head_pose(image_points)
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
    if yaw > 3:
        head=head+'right '
    elif yaw < -3: 
        head=head+'left '
    if pitch > 10:
        head=head+'up'
    elif pitch < 1:
        head=head+'down'
    
    alert(disappear_frame,stranger_frame)
    # print(stranger_cnt)
    draw_object_detection(boxes, scores, classes, LABELS)
    draw_head_pose(pitch, yaw, head)
    draw_face_reg(faces)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
print(face_loc)