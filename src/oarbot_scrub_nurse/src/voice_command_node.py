#!/usr/bin/env python3
# What this does:
#   - Listens for voice commands using Whisper
#   - Publishes recognized commands to the ROS topic /voice_command
#   - Any other ROS node (e.g. arm controller) can subscribe to
#     /voice_command and react when a command arrives
#
# To run (roscore must already be running):
#   rosrun oarbot_scrub_nurse voice_command_node.py
#
# To see what it's publishing (in a separate terminal):
#   rostopic echo /voice_command
# ============================================================

# rospy: the ROS library for Python. Lets this script talk to ROS.
import rospy

# String: the ROS message type we'll publish.
from std_msgs.msg import String

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav


# ---- SETTINGS ----------------------------------------

WHISPER_MODEL = "tiny"
RECORD_SECONDS = 3
SAMPLE_RATE = 16000
COMMANDS = ["scalpel", "scissors", "forceps", "ready", "return", "stop"]


# ---- FUNCTIONS -----------------------

def listen():
    """Records audio from microphone, returns numpy float32 array."""
    rospy.loginfo("Listening...")  # rospy.loginfo = ROS version of print()
                                   # shows up in terminal AND in ROS logs

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.squeeze()


def save_and_transcribe(audio):
    """Saves audio to temp file, transcribes with Whisper, returns text."""
    wav.write("/tmp/cmd.wav", SAMPLE_RATE, audio)
    result = model.transcribe("/tmp/cmd.wav")
    return result["text"]


def match_command(text):
    """Checks transcribed text for known commands. Returns command or None."""
    text = text.lower().strip()
    for cmd in COMMANDS:
        if cmd in text:
            return cmd
    return None


# ---- MAIN FUNCTION ------------------------------------------

def main():
    # --- ROS SETUP ---

    # Register this script as a ROS node named 'voice_command_node'
    # anonymous=False means only one instance of this node can run at a time
    rospy.init_node('voice_command_node', anonymous=False)

    # Create a publisher object that can send messages to the /voice_command topic
    # Arguments:
    #   '/voice_command' → the topic name (channel) to broadcast on
    #                      any node that wants to hear commands subscribes to this
    #   String           → the message type (we're sending plain text)
    #   queue_size=10    → if nothing is listening, hold up to 10 messages
    #                      in a buffer before dropping them
    pub = rospy.Publisher('/voice_command', String, queue_size=10)

    # Tell ROS how fast this node should run (10 times per second maximum)
    # This doesn't control recording, just prevents the loop from
    # running faster than necessary and wasting CPU
    rate = rospy.Rate(10)

    rospy.loginfo("Voice command node started. Listening for commands...")

    # --- MAIN LOOP ---

    # rospy.is_shutdown() returns True when someone kills the node (Ctrl+C)
    # ROS-safe while loop that will exit cleanly when the node is shut down
    while not rospy.is_shutdown():

        # Step 1: Record audio
        audio = listen()

        # Step 2: Transcribe to text
        text = save_and_transcribe(audio)
        rospy.loginfo(f"Heard: '{text}'")

        # Step 3: Check for known commands
        command = match_command(text)

        if command:
            rospy.loginfo(f"Publishing command: '{command}'")

            # pub.publish() sends the message out on /voice_command
            # String(command) wraps our plain Python string into a ROS String message
            # Any node subscribed to /voice_command will now receive this
            pub.publish(String(command))

        else:
            rospy.loginfo("No command recognized")

        # rate.sleep() waits the right amount of time to maintain our loop rate
        # Without this the loop would spin as fast as possible and waste CPU
        rate.sleep()


# ---- ENTRY POINT --------------------------------------------

# This is standard Python — only run main() if this file is run directly,
# not if it's imported by another script
if __name__ == '__main__':
    try:
        # Load whisper model once before entering the main loop
        # Loading inside the loop would be very slow (reloads every 3 seconds)
        rospy.loginfo(f"Loading Whisper '{WHISPER_MODEL}' model...")
        model = whisper.load_model(WHISPER_MODEL)
        rospy.loginfo("Model loaded.")

        main()

    except rospy.ROSInterruptException:
        # This catches the Ctrl+C shutdown signal from ROS
        # Lets the node exit cleanly instead of showing an error
        pass
