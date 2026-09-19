import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera niet beschikbaar")
   

while True:
    success, frame = camera.read()

    if not success:
        print("Kon geen frame lezen")
        break

    cv2.imshow("Smart Security Camera", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()