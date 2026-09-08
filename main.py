import argparse
import sys
from src.predict import SpamDetectorCLI

def main():
    parser = argparse.ArgumentParser(
        description="NLP-Based Intelligent Spam Detection System (CLI Executable)"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True, 
        help="Input text message to classify as Spam or Ham."
    )
    
    args = parser.parse_args()
    
    try:
        detector = SpamDetectorCLI()
        label, confidence = detector.classify_text(args.text)
        
        print("=" * 50)
        print(" SPAM DETECTION RESULT ")
        print("=" * 50)
        print(f"Input Message : {args.text}")
        print(f"Classification: {label}")
        print(f"Confidence    : {confidence:.2f}%")
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
