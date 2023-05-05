#Author: Eleni Jackson
#Date:5/5/2023
#Description:
#   This program utilizing media pipe hands to detect hand gestures - specifically ones that you use to play rock, paper, scissors.
#   Once running the player can play with two hands (two of their own or one of theirs and one of another player)
#   or they are able to play with one hand - while the second 'player' is the computer generating a random
#   move each time. If hands are unidentifiable you will get a kind error message,
#   otherwise you will find out who won and can continue playing a round every 150 frames.

#imports
import cv2 as cv
import mediapipe as mp
import random

#mp variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#identify what the player's move is
def getPlayersHandMove(hand_landmarks):
    #specific point on hand as an object
    landmarks = hand_landmarks.landmark
    #check for rock - tip of finger landmark should fall under bottom of finger landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(9,20,4)]): return "rock"
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y: return "scissor"
    else: return "paper"

#if there is only one hand - have the commputer play as player 2
def getComputerHandMove():
    #pick and return random option out of array
    return random.choice(["rock", "paper", "scissor"])

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

#clock to manage the game
clock = 0
p1_move = None
p2_move = None
text = ""
success = True

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = video.read()
        if not ret or frame is None: break

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        frame = cv.flip(frame, 1)

        if 0 <= clock < 20:
            success = True
            text = "Ready to play?"
        elif clock < 30: text = "Rock..."
        elif clock < 50: text = "Paper..."
        elif clock < 70: text = "Scissor..."
        elif clock < 90: text = "Shoot!"
        elif clock == 92:
            #Proceed with actual hand detection here
            player_hands = results.multi_hand_landmarks
            #two players - detect two hands
            if player_hands and len(player_hands) == 2:
                p1_move = getPlayersHandMove(player_hands[0])
                p2_move = getPlayersHandMove(player_hands[1])
            #one player - have computer choose as player 2
            elif player_hands and len(player_hands) == 1:
               p1_move = getPlayersHandMove(player_hands[0])
               p2_move = getComputerHandMove()
            #else - something went wrong
            else:
                success = False
        elif clock < 150:
            if success:
                text = f"Player 1 played {p1_move}, Player 2 played {p2_move}"
                if p1_move == p2_move: text = f"{text} Tie!"
                elif p1_move == "rock" and p2_move == "scissor": text = f"{text} Player 1 wins!"
                elif p1_move == "paper" and p2_move == "rock": text = f"{text} Player 1 wins!"
                elif p1_move == "scissor" and p2_move == "paper": text = f"{text} Player 1 wins!"
                else: text = f"{text} Player 2 wins!"
            else:
                text = "Uh-Oh - there was an issue!"

        #display the clock on window
        cv.putText(frame, f"Clock: {clock}", (50,50), cv.FONT_HERSHEY_PLAIN, 2, (76,153,0), 2, cv.LINE_AA)
        #display the game's text
        cv.putText(frame, text, (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (76, 153, 0), 2, cv.LINE_AA)

        #update clock
        clock = (clock + 1) % 150

        # imshow to display image in window 'frame'
        cv.imshow('frame', frame)
        # break out of loop when you hit q on keyboard
        if cv.waitKey(1) & 0xFF == ord('q'): break

#stop video and close windows
video.release()
cv.destroyAllWindows()