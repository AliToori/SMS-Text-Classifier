import subprocess
import os
import random
import time
import platform
import pyautogui
import logging.config
from pathlib import Path
import sys

class GitAutomation:
    """
    A class to automate Git operations for README.md changes in PyCharm,
    including undo/redo actions, staging, committing, and pushing.
    """

    def __init__(self, repo_path: str = None):
        """
        Initialize the GitAutomation class with repository settings and configurations.

        :param repo_path: Path to the Git repository directory. If None, uses the script's directory.
        """
        # Step 1: Set repository path
        if repo_path is None:
            # Use the directory containing this script
            self.repo_path = Path(os.path.abspath(os.path.dirname(__file__)))
        else:
            # Use provided path
            self.repo_path = Path(repo_path).resolve()

        # Validate repository path
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path {self.repo_path} does not exist.")
        if not (self.repo_path / '.git').is_dir():
            raise ValueError(f"Directory {self.repo_path} is not a Git repository.")
        os.chdir(self.repo_path)

        # Step 2: Define sequence for number of undo/redo actions
        self.sequence = [2, 3, 5, 7, 9, 1]

        # Step 3: Detect OS for correct control key
        self.is_mac = platform.system() == 'Darwin'
        self.ctrl_key = 'cmd' if self.is_mac else 'ctrl'

        # Step 4: Configure pyautogui
        pyautogui.FAILSAFE = True  # Move mouse to top-left to abort if needed
        pyautogui.PAUSE = 0.2  # Delay between actions

        # Step 5: Initialize logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Configure and return a logger for the Git automation process.

        :return: Configured Logger instance.
        """
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "[%(asctime)s,%(lineno)s] [%(message)s]",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "filename": "git_automation.log",
                    "maxBytes": 5 * 1024 * 1024,  # 5 MB
                    "backupCount": 3,
                    "encoding": "utf-8"
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "file"]
            },
        })
        return logging.getLogger()

    def _focus_pycharm(self) -> bool:
        """
        Attempt to focus the PyCharm window based on the operating system.

        :return: True if focus succeeded, False otherwise.
        """
        system = platform.system()
        self.logger.info(f"Attempting to focus PyCharm on {system}...")
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
                self.logger.error("Unsupported OS for window focus automation.")
                return False
            self.logger.info("PyCharm window focused successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to focus PyCharm: {e}. Please ensure PyCharm is open.")
            return False

    def _execute_git_command(self, command: list, action: str) -> bool:
        """
        Execute a Git command and handle errors.

        :param command: List of command components (e.g., ['git', 'add', '.']).
        :param action: Description of the action for logging (e.g., "add").
        :return: True if command succeeded, False otherwise.
        """
        self.logger.info(f"Executing git {action}...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Error in git {action}: {result.stderr}")
            return False
        self.logger.info(f"Git {action} completed successfully.")
        return True

    def automate_commits(self, num_commits: int, base_commit_msg: str):
        """
        Automate the process of performing undo/redo actions, staging, and committing.

        :param num_commits: Number of commits to perform.
        :param base_commit_msg: Base message for commits.
        """
        self.logger.info(f"Starting automation for {num_commits} commits with message: {base_commit_msg}")

        for i in range(num_commits):
            self.logger.info(f"Processing commit {i+1}/{num_commits}")

            # Get number of actions from cycling sequence
            N = self.sequence[i % len(self.sequence)]

            # Alternate between undo and redo
            is_undo = (i % 2 == 0)
            action_type = "undo" if is_undo else "redo"
            hotkey_combo = [self.ctrl_key, 'z'] if is_undo else [self.ctrl_key, 'shift', 'z']

            self.logger.info(f"Automating {N} {action_type} actions in PyCharm...")

            # Focus PyCharm window
            if not self._focus_pycharm():
                input("Please focus the PyCharm README.md editor window and press Enter to continue...")
                time.sleep(0.5)

            # Perform undo/redo actions
            for _ in range(N):
                pyautogui.hotkey(*hotkey_combo)
                time.sleep(0.2)  # Delay to avoid overwhelming the editor

            # Save the file
            self.logger.info("Saving README.md in PyCharm...")
            pyautogui.hotkey(self.ctrl_key, 's')
            time.sleep(0.5)  # Wait for save

            # Stage changes (git add .)
            if not self._execute_git_command(['git', 'add', '.'], "add"):
                continue

            # Commit changes
            commit_msg = f"{base_commit_msg} (automated {action_type} {N} times)"
            self.logger.info(f"Committing with message: {commit_msg}")
            if not self._execute_git_command(['git', 'commit', '-m', commit_msg, '--allow-empty'], "commit"):
                continue

    def push_to_remote(self):
        """
        Prompt the user to push commits to the remote repository (origin main).
        """
        push_confirm = input("\nAll changes committed. Push to origin main? (y/n): ").lower()
        if push_confirm == 'y':
            self.logger.info("Pushing to origin main...")
            if self._execute_git_command(['git', 'push', '-u', 'origin', 'main'], "push"):
                self.logger.info("Pushed to origin main.")
            else:
                self.logger.error("Push to origin main failed.")
        else:
            self.logger.info("Push skipped.")

    def run(self):
        """
        Main method to run the Git automation process.
        """
        try:
            # Get user inputs
            num_commits = int(input("How many commits do you want to make before pushing? "))
            if num_commits < 1:
                raise ValueError("Number of commits must be at least 1.")
            base_commit_msg = input(
                "Enter the base commit message (or press Enter for default 'Update README.md'): "
            ) or "Update README.md"

            # Run automation
            self.automate_commits(num_commits, base_commit_msg)
            self.push_to_remote()

        except ValueError as e:
            self.logger.error(f"Invalid input: {e}. Defaulting to 1 commit.")
            self.automate_commits(1, "Update README.md")
            self.push_to_remote()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Initialize and run the automation
    git_automation = GitAutomation()  # Defaults to script's directory
    # Or specify path explicitly: GitAutomation(repo_path='E:\\AliBots\\HealthcareCostsPrediction')
    git_automation.run()