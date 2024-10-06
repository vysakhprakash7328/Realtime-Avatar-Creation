import cv2
import numpy as np
import python_avatars as pa

def extract_average_color(image, x, y, width, height):
    """Extract the average color from a region of the image."""
    region = image[y:y+height, x:x+width]
    average_color = np.mean(region, axis=(0, 1))
    return average_color


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def detect_glasses(face_img):
    """Detect if the person is wearing glasses based on eye detection."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


    
    if len(eyes) > 0:
        return False
    else:
        return True





def detect_facial_hairs(face_img):
    """Detect if there is facial hair present using color detection."""
    # Convert the image to HSV for better color detection
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

    # Define range for skin color (for example purposes, adjust as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask for skin
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Check if there is a significant amount of non-skin color in the chin area
    chin_region = skin_mask[int(face_img.shape[0]*0.6):, :]  # Lower part of the face
    facial_hair_ratio = np.sum(chin_region == 0) / (chin_region.size)  # Calculate ratio of non-skin pixels

    return facial_hair_ratio > 0.1  # If more than 10% of the region is non-skin, consider it has facial hair


def detect_gender(face_rect, face_img):
    """Detect gender based on face size and facial hair presence."""
    (x, y, w, h) = face_rect

    # Heuristic for determining if it's a child
    if h < 100:  # Example threshold for height to determine if it's a child
        return "Child"

    # Check for facial hair
    has_facial_hair = detect_facial_hairs(face_img[y:y+h, x:x+w])

    if has_facial_hair:
        return "Man"
    else:
        return "Woman"
    



def detect_skin_color(image):
    """Approximate skin color from the image using a face detection region."""
    height, width, _ = image.shape
    skin_region = image[int(height * 0.35):int(height * 0.6), int(width * 0.3):int(width * 0.7)]
    average_color = np.mean(skin_region, axis=(0, 1))  # Average color of skin region
    return average_color

def detect_hair_type( gender):
    """Determine hair type based on color and possibly other heuristics."""
    if gender == "Man":
        return pa.HairType.SHORT_WAVED
    elif gender == "Women":
        return pa.HairType.BIG_HAIR
    else:
        return pa.HairType.SHORT_ROUND
    

def detect_facial_hair(skin_color_rgb):
    """Determine if there is facial hair present."""
    if skin_color_rgb[0] < 150:
        return pa.FacialHairType.BEARD_LIGHT  
    return pa.FacialHairType.NONE

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces

def extract_features(img, gender):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hair_color_rgb = extract_average_color(img_rgb, 50, 50, 100, 50)
    
    skin_color_rgb = detect_skin_color(img_rgb)

    hair_color_hex = "#{:02x}{:02x}{:02x}".format(int(hair_color_rgb[0]), int(hair_color_rgb[1]), int(hair_color_rgb[2]))
    skin_color_hex = "#{:02x}{:02x}{:02x}".format(int(skin_color_rgb[0]), int(skin_color_rgb[1]), int(skin_color_rgb[2]))
    detect_glass = detect_glasses(img)
    hair_type = detect_hair_type( gender)
    facial_hair = detect_facial_hair(skin_color_rgb)

    return {
        "hair_type": hair_type,
        "hair_color": hair_color_hex,
        "facial_hair": facial_hair,
        "mouth": pa.MouthType.SMILE,
        "eye_type": pa.EyeType.DEFAULT,
        "eyebrow_type": pa.EyebrowType.DEFAULT_NATURAL,
        "nose_type": pa.NoseType.DEFAULT,
        "skin_color": skin_color_hex,
        "glasses": pa.AccessoryType.SUNGLASSES if detect_glass == True else pa.AccessoryType.NONE
    }
    





def create_avatar(features):
    """Create an avatar using the extracted features."""
    my_custom_avatar = pa.Avatar(
        style=pa.AvatarStyle.CIRCLE,
        background_color=pa.BackgroundColor.WHITE,
        top=features['hair_type'],
        eyebrows=features['eyebrow_type'],
        eyes=features['eye_type'],
        nose=features['nose_type'],
        mouth=features['mouth'],
        facial_hair=features['facial_hair'],
        skin_color=features['skin_color'],
        hair_color=features['hair_color'],
        accessory=features['glasses'],
        clothing=pa.ClothingType.HOODIE,
        clothing_color=pa.ClothingColor.BLACK
    )

    avatar_image = my_custom_avatar.render("my_avatar.svg")
    

    
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    print("Press 'c' to capture an image, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture image
            img = frame.copy()  # Make a copy of the current frame
            print("Image saved as 'captured_image.png'")
            break
        elif key == ord('q'):  # Quit
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    faces = detect_faces(img)
    if len(faces) == 0:
        print("No faces detected.")
        return

    gender = detect_gender(faces[0], img)
    print(f"Detected gender: {gender}")


    features = extract_features(img, gender)

    create_avatar(features)

    


if __name__ == '__main__':
    main()
