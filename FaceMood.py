import tempfile
import cv2 as cv
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf

from PIL import Image
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

# emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load the cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def get_models():
    model_type = st.sidebar.selectbox('Model', ('Model 1', 'Model 2'))
    
    # load emotion model
    if model_type == 'Model 1':
        return tf.keras.models.load_model('./models/base_1_overfit.h5')
    else:
        return tf.keras.models.load_model('./models/emotion_recognizer.h5')


def opencv_detection(image, model):
    
    out_img = image.copy()
    
    # Detect the faces
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(out_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
        # Crop image to face
        print(x, y, x+w, y+h)
        cimg = image[y:y+h, x:x+w]
        cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
        
        # get model prediction
        pred = model.predict(cropped_img)
        idx =  np.argmax(pred)

        cv.putText(out_img, emotion_dict[idx], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)

        if mode == 'With cropped image':
            st.write('Emotion: ', emotion_dict[idx])
            st.image(cv.resize(cimg, (300, 300)), channels="BGR", caption='Cropped Image')
    

    if mode == 'With full image':
        st.image(out_img, channels="BGR", use_column_width=True)


def mediapipe_detection(image, model):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=detection_confidence) as face_detection:
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    image_rows, image_cols, _ = image.shape
    out_img = image.copy()

    if results.detections:
        for detection in results.detections:
            try:
                # Draw face detection box
                mp_drawing.draw_detection(out_img, detection)
                
                box = detection.location_data.relative_bounding_box
                x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols, image_rows)
                
                # Crop image to face
                cimg = image[x[1]-20:y[1]+20, x[0]-20:y[0]+20]
                cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
                
                # get model prediction
                pred = model.predict(cropped_img)
                idx =  np.argmax(pred)

                cv.putText(out_img, emotion_dict[idx], (x[0], x[1]-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)

                if mode == 'With cropped image':
                    st.write('Emotion: ', emotion_dict[idx])
                    st.image(cv.resize(cimg, (300, 300)), channels="BGR", caption='Cropped Image')
            
            except:
                pass

        if mode == 'With full image':
            st.image(out_img, channels="BGR", use_column_width=True)


# Basic App Scaffolding
st.title('Facial Emotion Recognition')

# Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 330px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 330px
        margin-left: -350px
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div:first-child h1{
        padding: 0rem 0rem 0rem 0rem;
        text-align: center;
        font-size: 2rem;
    }
    .css-1544g2n.e1fqkh3o4 {
        padding-top: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create Sidebar
st.sidebar.title('FaceMood')
st.sidebar.divider()

# Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'Page',
    ('About', 'Image', 'Video')
)


# About Page
if app_mode == 'About':
    st.markdown('''
                ## Face Mood \n
                In this application we are using **MediaPipe** for the Face Detection.
                **Tensorflow** is to create the Facial Emotion Recognition Model.
                **StreamLit** is to create the Web Graphical User Interface (GUI) \n
                
                - [Github](https://github.com/pc7122) \n
    ''')
    
    
# Image Page
elif app_mode == 'Image':

    # Create Sidebar
    model = get_models()
    mode = st.sidebar.radio('Mode', ('With full image', 'With cropped image'))
    detection_type = st.sidebar.radio('Detection Type', ['Mediapipe', 'OpenCV'])
    st.sidebar.divider()
    
    if detection_type == 'Mediapipe':
        detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.divider()
        
    img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    
    # Check if image is uploaded or not
    if img_file_buffer is not None:
        # read uploaded image
        image = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv.imdecode(image, 1)
    else:
        # read demo image
        demo_image = "assets/single face 2.jpg"
        image = cv.imread(demo_image)

    # Display Original Image on Sidebar
    st.sidebar.write('Original Image')
    st.sidebar.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), use_column_width=True)
    
    
    if detection_type == 'Mediapipe':
        mediapipe_detection(image, model)
    else:
        opencv_detection(image, model)    
        
        
# Video Page
elif app_mode == 'Video':

    model = get_models()
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.checkbox('Use Webcam')
    st.sidebar.divider()

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.divider()
    
    # Get Video
    stframe = st.image("assets/multi face.jpg", use_column_width=True)
    
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file_buffer:
        if use_webcam:
            video = cv.VideoCapture(0)
        else:
            video = None
    else:
        temp_file.write(video_file_buffer.read())
        video = cv.VideoCapture(temp_file.name)
        
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=detection_confidence) as face_detection:
        while use_webcam:
            ret, frame = video.read()
            image = frame.copy()
            
            if not ret:
                print("Ignoring empty camera frame.")
                video.release()
                break
            
            img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detection.process(img)
            
            image_rows, image_cols, _ = frame.shape
            
            if results.detections:
                for detection in results.detections:
                    try:
                        box = detection.location_data.relative_bounding_box

                        x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                        y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols,image_rows)

                        # Draw face detection box
                        mp_drawing.draw_detection(image, detection)
                        
                        # Crop image to face
                        cimg = frame[x[1]:y[1], x[0]:y[0]]
                        cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
                        
                        # get model prediction
                        pred = model.predict(cropped_img)
                        idx =  np.argmax(pred)
                        
                        image = cv.flip(image, 1)
                        cv.putText(image, emotion_dict[idx], (image_rows-x[0], x[1]-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                        image = cv.flip(image, 1)
                    
                    except:
                        print("Ignoring empty camera frame.")
                        pass
                
            stframe.image(cv.flip(image, 1), channels="BGR", use_column_width=True)
               
        if video is not None:
            video.release()

