import time
import pyautogui
import keyboard

def click_loop():
    while True:
        if keyboard.is_pressed('w'):  # Stop on '2'
            print("Stopped clicking")
            break
        pyautogui.click()  # Perform the click
        time.sleep(15)  # Wait for 15 seconds

def main():
    print("Press '1' to start, '2' to stop, '3' to exit.")
    while True:
        if keyboard.is_pressed('q'):
            print("Started clicking")
            click_loop()
        elif keyboard.is_pressed('e'):  # Exit on '3'
            print("Exiting program")
            break
        time.sleep(0.1)  # Small delay to prevent CPU overuse

if __name__ == "__main__":
    main()
