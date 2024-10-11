from deepface import DeepFace
import cv2
import numpy


SIZE = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, SIZE[0])
cap.set(4, SIZE[1])

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        continue

    cv2.flip(frame, 1, frame)

    for demography in DeepFace.analyze(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        ("gender", "age"),
        False,
        "mediapipe",
        silent=True,
    ):
        if demography["region"]["w"] != frame.shape[1]:
            box = numpy.int32(list(demography["region"].values())[:4])
            box[1] -= 20

            text = f"{'Male' if demography['dominant_gender'][0] == 'M' else 'Female'} {demography['age']}"

            cv2.rectangle(
                frame,
                box[:2],
                box[:2] + box[2:],
                (
                    (201, 217, 0)
                    if demography["dominant_gender"][0] == "M"
                    else (180, 105, 255)
                ),
                3,
            )

            (width, height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_COMPLEX, 1, 2
            )

            cv2.rectangle(
                frame,
                (box[0], box[1] - (height + baseline) - 20),
                (box[0] + width + 60, box[1] - 10),
                (
                    (201, 217, 0)
                    if demography["dominant_gender"][0] == "M"
                    else (180, 105, 255)
                ),
                -1,
            )

            cv2.putText(
                frame,
                text,
                (box[0] + 30, box[1] - baseline // 2 - 15),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow("Age and Gender Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()

cv2.destroyAllWindows()
