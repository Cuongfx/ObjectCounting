<img width="1800" alt="Image" src="https://github.com/user-attachments/assets/84077af1-04c4-4014-a4d0-76e7baa8ba35" /># Region-Based Human Detection & Tracking (YOLOv11)

## Overview
In many computer vision applications, it is crucial to count how many people enter or remain within certain zones in a scene (e.g., security areas, store aisles, event spaces). This project uses:

- **YOLOv11** for human (person) detection.  
- **Built-in object tracking** to maintain a consistent ID for each person across frames.  
- **Shapely** or a similar geometry library to define polygons (regions) and check if a person’s “foot point” enters or leaves a region.  
- **Real-time drawing** of bounding boxes, IDs, and region overlays on the video feed.

## Features
1. **Detection**: Uses YOLOv11 to detect people in each frame of a video.  
2. **Tracking**: Assigns unique IDs to each person, allowing accurate counting even if a person leaves and re-enters the region.  
3. **Region-based Counting**: Tracks how many unique IDs enter each polygon-defined region.  
4. **Visualization**: Draws bounding boxes, tracking IDs, region polygons, and counts.

## Requirements and Installation
1. **Python 3.8+** recommended.  
2. **CUDA toolkit** (optional) for GPU acceleration if you have a compatible NVIDIA GPU.  


