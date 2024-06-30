
# Face Measurement for Online Glasses Ordering

## Overview

With the surge in online shopping, especially for glasses, a common issue is the incorrect sizing of frames and lenses due to inaccurate self-measurements by customers. This often leads to returns, resulting in an increased carbon footprint. This project aims to mitigate this problem by providing a tool that accurately measures facial dimensions using a webcam, helping customers to choose the correct frame size and lens size, thereby reducing the return rate and contributing to a greener environment.

## Features

- **Real-time Face Landmark Detection**: Utilizes MediaPipe to detect facial landmarks in real-time from a webcam feed.
- **Distance Estimation**: Trains a linear regression model to estimate the distance of the face from the camera using bounding box dimensions.
- **Dynamic Glasses Overlay**: Draws glasses on the face dynamically scaled based on the detected measurements.
- **Measurement Display**: Displays real-time measurements of eye width, eye distance, and face width on the screen.

## Demo

![Demo GIF](demo/test.gif)

## Getting Started

### Prerequisites

- Python 3.7+
- Webcam

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/face-measurement-for-glasses-ordering.git
    cd face-measurement-for-glasses-ordering
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the main script:
    ```sh
    python src/main.py
    ```

2. The application will start the webcam feed and display real-time measurements and an overlay of glasses on your face.

## Detailed Description of Key Components

### Data Preparation and Model Training

**Calibration Data**: The dataset used for camera calibration contains bounding box sizes and corresponding distances, which helps train the linear regression model. This model estimates how far the user is from the camera, allowing dynamic scaling of glasses size.

```python
# Data format: bounding_boxes (array of [height, width]), distances (array of distances)
bounding_boxes = np.array([
    [135, 114], [200, 185], [113, 104], [185, 154], [360, 270], [280, 220],
    [164, 145], [245, 215], [430, 330], [135, 110], [104, 90], [95, 83],
    [88, 75], [78, 68], [193, 160], [297, 232]
])
distances = np.array([58, 36, 70, 45, 20, 28, 50, 30, 17, 60, 80, 90, 100, 110, 40, 25])

# Train the linear regression model
model = LinearRegression()
model.fit(bounding_boxes, distances)
```

### Real-time Processing

**Face Landmark Detection**: Uses MediaPipe to detect facial landmarks in real-time.

**Distance Estimation**: Applies the trained linear regression model to estimate the distance of the face from the camera.

**Measurement Calculation**: Calculates key facial dimensions like eye width, eye distance, and face width, which are crucial for determining the right frame and lens size.

### Dynamic Glasses Overlay

**Drawing Functions**: Functions to draw the glasses dynamically on the user's face, including dashed lines for measurement indicators.

```python
def draw_dashed_line(img, start, end, color, thickness, dash_length=5):
    x1, y1 = start
    x2, y2 = end
    line_length = calculate_distance(start, end)
    num_dashes = int(line_length // dash_length)
    for i in range(num_dashes):
        start_x = int(x1 + (x2 - x1) * (i / num_dashes))
        start_y = int(y1 + (y2 - y1) * (i / num_dashes))
        end_x = int(x1 + (x2 - x1) * ((i + 1) / num_dashes))
        end_y = int(y1 + (y2 - y1) * ((i + 1) / num_dashes))
        if i % 2 == 0:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
```

## Contributions

Feel free to open issues or submit pull requests if you have suggestions for improving this project.

## License

This project is licensed under the MIT License.
