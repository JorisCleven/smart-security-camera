import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import numpy as np
#--- 
import json
from datetime import datetime
from pathlib import Path
from time import monotonic


model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(0)
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_analyzer.prepare(ctx_id=-1)

reference_image = cv2.imread("data/known_person/testpersoon.jpg")

if reference_image is None:
    raise FileNotFoundError("Referentieafbeelding niet gevonden.")

#zoek gezichten in afbeelding
reference_faces = face_analyzer.get(reference_image)

if len(reference_faces) != 1:
    raise ValueError("Er moet precies één gezicht in de referentieafbeelding zijn.")

known_embedding = reference_faces[0].normed_embedding

if known_embedding is None:
    raise ValueError("kon geen embedding maken")

MATCH_THRESHOLD = 0.65

#---
#maak map voor opgeslagen embeddings
embedding_folder = Path("data/embeddings")
embedding_folder.mkdir(parents=True, exist_ok=True)

#bewaar embedding van testpersoon
np.save(
    embedding_folder / "testpersoon.npy",
    known_embedding
)

#instellingen voor eventopslag
EVENT_COOLDOWN_SECONDS = 15
last_saved_time = None
#---
# check camera 

if not camera.isOpened():
    raise RuntimeError("Camera kon niet worden geopend.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Kon geen frame lezen.")
            break

        faces = face_analyzer.get(frame)

        # einde camera check

        # start detectie

        results = model(frame, verbose=False)
        result = results[0]

        person_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name != "person":
                continue

            person_count += 1

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

        detected_names = []
        #verwerk gevonden gezichten loop
        for face in faces:
            # pak gezicht coords
            bbox = face.bbox.astype(int)

            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # standaard naam label
            name = "Unknown"
            label = name

            # haal de embedding van het live gezicht op
            current_embedding = face.normed_embedding

            # vergelijk de live embedding met de referentie
            if current_embedding is not None:
                similarity = float(
                    np.dot(known_embedding, current_embedding)
                )

                if similarity >= MATCH_THRESHOLD:
                    name = "Testpersoon"

                label = f"{name}: {similarity:.2f}"

            detected_names.append(name)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        current_time = monotonic()

        enough_time_passed = (
            last_saved_time is None
            or current_time - last_saved_time >= EVENT_COOLDOWN_SECONDS
        )

        detection_found = person_count > 0 or len(faces) > 0

        if detection_found and enough_time_passed:
            event = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "person_count": person_count,
                "face_count": len(faces),
                "recognized_faces": detected_names
            }

            event_file_path = Path("data/events.jsonl")

            with event_file_path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(event) + "\n")

            last_saved_time = current_time
            print("detectie opgeslagen:")

        cv2.imshow("Smart Security Camera", frame)

        # check voor q key & stop 
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

#einde check