"""
Hand FX – main loop
Run: python main.py
Press 'q' to quit, '1'/'2'/'3' to switch effects.
"""

import cv2
from detector import HandDetector
from effects.drawing import AirDrawEffect
from effects.particles import ParticleEffect
from effects.gestures import GestureEffect

# dict for type of effects
EFFECTS = {
    ord("1"): ("Air Drawing", AirDrawEffect),
    ord("2"): ("Particles", ParticleEffect),
    ord("3"): ("Gestures", GestureEffect),
}

def main():
    # setting up camera and hand detector
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    active_name, ActiveClass = "Air Drawing", AirDrawEffect
    effect = ActiveClass()

    print("Keys: 1 = Air Drawing  2 = Particles  3 = Gestures  q = Quit")

    # loop to read frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        hands = detector.find_hands(frame) 
        frame = effect.apply(frame, hands)

        # HUD
        cv2.putText(frame, f"Effect: {active_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "1 Draw | 2 Particles | 3 Gestures | q Quit",
                    (20, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hand FX", frame)

        # change mode if detected
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in EFFECTS:
            active_name, ActiveClass = EFFECTS[key]
            effect = ActiveClass()
            print(f"Switched to: {active_name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()