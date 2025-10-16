import subprocess
import os
import random
import time
import platform
import pyautogui


def focus_pycharm():
    """Attempt to focus the PyCharm window based on the OS."""
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['osascript', '-e', 'tell application "PyCharm" to activate'])
        elif system == 'Linux':  # Linux
            subprocess.run(['wmctrl', '-a', 'PyCharm'])
        elif system == 'Windows':  # Windows
            import win32gui
            import win32con
            def callback(hwnd, extra):
                if 'PyCharm' in win32gui.GetWindowText(hwnd):
                    win32gui.SetForegroundWindow(hwnd)
                    return False

            win32gui.EnumWindows(callback, None)
        else:
            print("Unsupported OS for window focus automation.")
            return False
        return True
    except Exception as e:
        print(f"Failed to focus PyCharm: {e}. Please ensure PyCharm is open.")
        return False


def main():
    # Change to the repository directory; assuming script is run from PyCharm in the repo directory
    repo_path = 'E:\AliBots\HealthcareCostsPrediction'
    os.chdir(repo_path)

    # Sequence for the number of undo/redo actions
    sequence = [2, 3, 5, 7, 9, 1]

    # Ask how many commits to make before pushing
    try:
        num_commits = int(input("How many commits do you want to make before pushing? "))
    except ValueError:
        print("Invalid input. Defaulting to 1 commit.")
        num_commits = 1

    # Ask for the base commit message
    base_commit_msg = input(
        "Enter the base commit message (or press Enter for default 'Update README.md'): ") or "Update README.md"

    # Detect OS for correct control key (Ctrl on Windows/Linux, Cmd on macOS)
    is_mac = platform.system() == 'Darwin'
    ctrl_key = 'cmd' if is_mac else 'ctrl'

    # Configure pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to top-left to abort if needed
    pyautogui.PAUSE = 0.2  # Delay between actions

    for i in range(num_commits):
        print(f"\nProcessing commit {i + 1}/{num_commits}")

        # Get N from cycling sequence
        N = sequence[i % len(sequence)]

        # Alternate between undo and redo: even i (0-based) = undo, odd = redo
        is_undo = (i % 2 == 0)
        action_type = "undo" if is_undo else "redo"
        hotkey_combo = [ctrl_key, 'z'] if is_undo else [ctrl_key, 'shift', 'z']

        print(f"Automating {N} {action_type} actions in PyCharm...")

        # Focus PyCharm window
        if not focus_pycharm():
            input("Please focus the PyCharm README.md editor window and press Enter to continue...")

        # Short delay to ensure focus
        time.sleep(0.5)

        # Perform the undo/redo actions
        for _ in range(N):
            pyautogui.hotkey(*hotkey_combo)
            time.sleep(0.2)  # Delay to avoid overwhelming the editor

        # Save the file (Ctrl+S or Cmd+S)
        pyautogui.hotkey(ctrl_key, 's')
        time.sleep(0.5)  # Wait for save to complete

        # Stage all changes (git add .)
        result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in git add: {result.stderr}")
            continue

        # Commit with message
        commit_msg = base_commit_msg
        print(f"Committing with message: {commit_msg}")
        result = subprocess.run(['git', 'commit', '-m', commit_msg, '--allow-empty'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in git commit: {result.stderr}")
            continue

    # After all changes, ask to push
    push_confirm = input("\nAll changes committed. Push to origin main? (y/n): ").lower()
    if push_confirm == 'y':
        # Push with -u to set upstream if needed (assuming remote is 'origin' and branch is 'main')
        result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Pushed to origin main.")
        else:
            print(f"Error in git push: {result.stderr}")
    else:
        print("Push skipped.")


if __name__ == "__main__":
    main()