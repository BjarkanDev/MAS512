import subprocess
import sys

def main():
    
    # Run question 1
    subprocess.run(["python", "question1.py"])

    # Run question 3
    subprocess.run(["python", "question3.py"])

    return 0

if __name__ == "__main__":
    sys.exit(main())
