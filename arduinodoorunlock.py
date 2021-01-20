# arduino_door_user.py

import serial
import time

# Define the serial port and baud rate.
# Ensure the 'COM#' corresponds to what was seen in the Windows Device Manager
ser = serial.Serial('COM8', 9600)

def door_on_off(user_input):
    # user_input = input("\n Type on / off / quit : ")
    if user_input =="open":
        print("door is on...")
        time.sleep(0.1) 
        ser.write(b'H') 
        door_on_off()
    elif user_input =="close":
        print("door is off...")
        time.sleep(0.1)
        ser.write(b'L')
        door_on_off()
    # elif user_input =="quit" or user_input == "q":
    #     print("Program Exiting")
    #     time.sleep(0.1)
    #     ser.write(b'L')
    #     ser.close()
    else:
        print("Invalid input. Type on / off ")
        door_on_off()

time.sleep(2) # wait for the serial connection to initialize

