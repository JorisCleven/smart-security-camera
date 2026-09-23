import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(0)
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_analyzer.prepare(ctx_id=-1)

# check camera 

if not camera.isOpened():
    raise RuntimeError("Camera kon niet worden geopend.")

while True:
    success, frame = camera.read()

    if not success:
        print("Kon geen frame lezen.")
        break

    faces= face_analyzer.get(frame)

    # einde camera check
    
    # start detectie

    results = model(frame, verbose=False)
    result = results[0]
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name != "person":
            continue

        coordinates = box.xyxy[0].tolist()

        x1 = int(coordinates[0])
        y1 = int(coordinates[1])
        x2 = int(coordinates[2])
        y2 = int(coordinates[3])

        confidence = float(box.conf[0])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Persoon: {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    #verwerk gevonden gezichten
    for face in faces:
        bbox = face.bbox.astype(int)

        x1 =int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        #blauw: gezicht
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    cv2.imshow("Smart Security Camera", frame)

    # check voor q key & stop 

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

#einde check