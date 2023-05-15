# Pose_Estimation_Using_OpenCV
We have designed a Pose Estimation model with MediaPipe library and open-cv. In addition to Pose-Estimation it is capable of angle detection and stage detection. It follows a step-by-step approach helping you connect o the next part. Let's go.

# Real-time Pose Estimation-
Here we do real-time pose detection plus angle and stage detection with MediaPipe as it is a well-balanced framework doping real-time detection on the CPU.

# Media Pipe-
##### Single Person pose detection frameowrk
##### Follows Top-Down approach i.e. two-stage working with detection first and then tracking
##### As detection is not done in all frames hence inference is fast.
##### Follows Blaze Pose Topology as shown below
![image](https://github.com/SonakshiChauhan/Pose_Estimation_Using_OpenCV/assets/91408631/cc9891a1-3213-49f3-8fcf-7eb8216df106)

# Deliverables of Model:
The model will do real-time pose detection accompanied by angle detection, stage display, and count of stage change, as shown in the video below.

https://github.com/SonakshiChauhan/Pose_Estimation_Using_OpenCV/assets/91408631/940364ad-5162-497c-aa72-9f916acbf48f


# Topics Covered
1. Pose Estimation
2. Open-CV and MediaPipe usage
3. Logic for angle estimation
4. Stage and count description logic

# Tools
1. MediaPipe
2. Open CV
3. Google Colab
4. Numpy

# Installation
The following library needs to be downloaded before moving further
```bash
pip install mediapipe opencv-python
```
# Notebook Structure
1. Import
2. Video Capture
3. Make Detection
4. Determining Joints
5. Calculate Angles
6. Angle Calculation
7. Curl Counter

# 1. Import

```bash
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
```
• "mp_pose" acts as a variable holding the insatnce of mediapipe detetction model<br>
• "mp_drawing" using this we will draw the detected landmarks on our captured frame and video

# 2.Video Capture

```bash
#Video Capture
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    cv2.imshow("Media Pipe",frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
• Here we instructing real-time detection using webcam and storing in "cap". <br>
• Then till webcam is opened we access frame and view it.<br>
• As soon as key 'q' is pressed the capture action stops.

# 3.Make Detection
```bash
cap=cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        
        #Recolor the image to RGB
        image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        
        #make detection
        results=pose.process(image)
        
        #Recolor back to BGR
        image.flags.writeable=True
        image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Render the detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2))
        
        cv2.imshow("Media Pipe",image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```
• The difference in step 2 and 3 is the code for detecting<br>
• The frame captured is first changed from BGR to RGB.<br>
• Further we process pose detetcion on our image format using "process" function.<br>
• Then we draw the detected landmarks on the image
• Now we display the detections in realtime

# 4. Determining Joints
```bash
cap=cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        
        #Recolor the image to RGB
        image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        
        #make detection
        results=pose.process(image)
        
        #Recolor back to BGR
        image.flags.writeable=True
        image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks=results.pose_landmarks.landmark
            # print(landmarks)
        except:
            pass
        
        #Render the detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2))
        
        cv2.imshow("Media Pipe",image)
         
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```
• The difference between Step 3 and Step 4 is the code for detetcing Joints.<br>
• From the detected image we extract landmarks which are points depicting the joints.

# 5. Calculate Angles
```bash
#Claculate angle
def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians=np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle=360-angle
        
    return angle
```
• Above is the fucntion for calculating angle between the body points.<br>
• This done using numpy library

# 6.Angle Calculation
```bash
angle calculation 
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                       
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```
• Here is where the angle function we made in step 5 is used.<br>
• Using the joints we extracted in Step 4 we extract the coordinated of shoulder, elbow and wrist<br>
• After calculating angle using we vizulaize it by using the "cv2.putText" method.

# 7. Curl Counter
```bash
#curl counter
cap = cv2.VideoCapture(0)

#curl counter variables
counter=0
stage=None
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            #curl counter logic
            if angle>160:
                stage="down"
            if angle < 30 and stage=="down":
                stage="up"
                counter+=1
                print(counter)
                       
        except:
            pass
        
        #render curl counter
        #setup status box
        cv2.rectangle(image,(0,0),(225,73),(245,177,16),-1)
        
        #Rep data
        cv2.putText(image,"Reps",(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        
        #Stage data
        cv2.putText(image,"Stage",(65,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,(60,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)   
             
         
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
```
• Here we use the angle calculated and we use that to determine the stage(UP or DOWN) of the body part whose angle we are calculating( SHOULDER, ELBOW, and WRIST).<br>
• When the angle is 180 the hand (in our case) is in the down position, but keeping in mind the abnormalities we set it to 160 for down.<br>
• When the stage is down and the angle is 30 then we change it to stage "Up" and increase the counter once.<br>
• Then we display the stage and count results in a rectangle in the image.

## ✨Code Completed✨<br>
### Contact: 
📧: sonakshichauhan1402@gmail.com <br>
LinkedIn: Sonakshi Chauhan

## Project Continuity
This project is complete

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
