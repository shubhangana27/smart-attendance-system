# Smart Attendance System (Computer Vision CLI)

An automated, terminal-executable Smart Attendance System built using OpenCV, Python, and feature extraction algorithms to process image inputs and log attendance into a CSV database.

## 📌 Project Overview
- **Core Task:** Automated Face Recognition & Attendance Logging
- **Interface:** Command-Line Interface (CLI)
- **Tech Stack:** Python 3.14, OpenCV, NumPy, Pandas

---

## 🛠️ Setup & Installation

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

📂 Project Structure
smart-attendance-system/
├── known_faces/        # Known individual database profiles
│   └── shubhangana.jpg
├── test_images/        # Input test images for processing
│   └── sample.jpg
├── attendance.csv      # Log output generated dynamically
├── main.py             # Main CLI execution script
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

🚀 Execution Instructions
Run the system strictly via terminal using the following CLI arguments:
python main.py --known_dir known_faces --input test_images/sample.jpg --output attendance.csv

Options:
--known_dir: Path to folder containing target profiles (default: known_faces)
--input: Path to input test image for detection (required)
--output: Output CSV file path (default: attendance.csv)

**Step 7: Prepare Git & Push to GitHub**

1. Create a `.gitignore` file in your project folder and add this line so temporary files aren't uploaded:
   ```text
   venv/
   __pycache__/
   