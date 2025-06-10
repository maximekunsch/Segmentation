This is my Project of MI206. The Goal was to Implementing a segmentation of the eye's vascular network.

# Installation Guide for Image Segmentation Environment

This guide will help you set up a Python virtual environment and install the necessary libraries to run the image segmentation code.

## Prerequisites

- Python 3.6 or higher
- pip (usually comes installed with Python)

## Installation Steps

### 1. Create a Virtual Environment

Open a terminal and run the following command to create a virtual environment:

```bash
python -m venv my_env
```

Replace `my_env` with the name you want to give to your virtual environment.

### 2. Activate the Virtual Environment

- **On Windows:**

```bash
my_env\Scripts\activate
```

- **On macOS/Linux:**

```bash
source my_env/bin/activate
```

### 3. Install the Required Libraries

Once the virtual environment is activated, install the required libraries by running the following command:

```bash
pip install numpy scikit-image opencv-python matplotlib pillow
```

### 4. Run the Code

Ensure your virtual environment is activated, then run your Python script:

```bash
python your_script.py
```

Replace `your_script.py` with the name of your script file.

### 5. Deactivate the Virtual Environment

Once you are done, you can deactivate the virtual environment by running:

```bash
deactivate
```

## Troubleshooting

- Ensure that Python and pip are correctly installed and accessible from your terminal.
- If you encounter errors during the installation of libraries, try upgrading pip with `pip install --upgrade pip` before attempting again.
