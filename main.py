import cv2
from detector import HandDetector
from effects.drawing import AirDrawEffect
from effects.gestures import GestureEffect
from effects.bbox import BoundingBoxEffect

EFFECTS = {
    ord("1"): ("Air Drawing",    AirDrawEffect),
    ord("2"): ("BBox Filters",   BoundingBoxEffect),
    ord("3"): ("Gestures",       GestureEffect),
}


def main():
    # initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    active_name, ActiveClass = "Air Drawing", AirDrawEffect
    effect = ActiveClass()

    # read frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hands = detector.find_hands(frame)
        frame = effect.apply(frame, hands)

        cv2.putText(frame, f"Mode: {active_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "1 Draw | 2 Filters | 3 Gestures | q Quit",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Hand FX", frame)


        # check for keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key in EFFECTS:
            active_name, ActiveClass = EFFECTS[key]
            effect = ActiveClass()
            print(f"Switched to: {active_name}")

        # for bbox
        if isinstance(effect, BoundingBoxEffect):
            if key == ord("f"):
                effect.cycle_filter()
            elif key == ord("c"):
                effect.clear_filters()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()