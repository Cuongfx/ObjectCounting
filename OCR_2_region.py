import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from shapely.geometry import Point, Polygon
import os


class RegionPersistentCounter:
    def __init__(self, video_path, region_points, model_path="yolo11n.pt", output_path="region_counting.mp4"):
        self.video_path = video_path
        self.region_points = region_points
        self.region_polygon = Polygon(region_points)
        self.model_path = model_path
        self.output_path = output_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), f"Error reading video file: {video_path}"
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer - use mp4v codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        self.video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        # Initialize the model - stick with yolo11n.pt as requested
        try:
            self.model = YOLO(model_path)
            print(f"Model {model_path} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  # Stop execution if model can't be loaded
        
        # Tracking variables
        self.tracked_ids = {}  # Dictionary to store tracked IDs
        self.entry_count = 0   # Counter for unique entries
        self.colors = Colors() # For visualization
        
        # Debug variables
        self.class_counts = {}  # Count occurrences of each class
        
    def is_in_region(self, box):
        """Check if the bottom center of a bounding box is in the region"""
        x1, y1, x2, y2 = box
        foot_point = Point((x1 + x2) / 2, y2)
        return self.region_polygon.contains(foot_point)
    
    def draw_region(self, frame):
        """Draw the region on the frame"""
        pts = np.array(self.region_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw outline
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Fill with transparent color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw statistics on the frame"""
        # Draw total count
        cv2.putText(
            frame, 
            f"Total People Entered: {self.entry_count}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Draw IDs that have been counted
        y_pos = 80
        for track_id in self.tracked_ids:
            color = self.colors(int(track_id), True)
            cv2.putText(
                frame, 
                f"ID {track_id}", 
                (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
            y_pos += 30
            
        # If we have class count data in debug mode, show it
        if self.class_counts and len(self.class_counts) > 0:
            y_pos = 80
            for cls, count in self.class_counts.items():
                cv2.putText(
                    frame, 
                    f"Class {cls}: {count}", 
                    (self.width - 200, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                y_pos += 30
            
        return frame
    
    def process_frame(self, frame, frame_count):
        """Process a single frame"""
        # Make a copy for drawing
        output_frame = frame.copy()
        
        # Run detection and tracking - try without class filtering
        results = self.model.track(frame, persist=True, verbose=False)
        
        # Draw the region
        output_frame = self.draw_region(output_frame)
        
        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Extract all classes for debugging
            if hasattr(results[0].boxes, 'cls') and results[0].boxes.cls is not None:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Update class counts
                for cls in classes:
                    if cls not in self.class_counts:
                        self.class_counts[cls] = 0
                    self.class_counts[cls] += 1
                
                # Print detected classes in first few frames
                if frame_count <= 10:
                    unique_classes = np.unique(classes)
                    print(f"Frame {frame_count} - Detected classes: {unique_classes}")
            
            # Check if we have tracking IDs
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                try:
                    # Extract boxes and IDs
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    # Process all detections - try both class 0 and class 1 for people
                    for i, (box, cls, track_id) in enumerate(zip(boxes, classes, track_ids)):
                        # For the YOLO11n model, we'll try both class 0 (default person) and class 1
                        if cls == 0 or cls == 1:  # Try both common person class IDs
                            # Get box coordinates
                            x1, y1, x2, y2 = box
                            
                            # Check if in region
                            is_in_region = self.is_in_region(box)
                            
                            # Count if in region and not counted before
                            if is_in_region and track_id not in self.tracked_ids:
                                self.tracked_ids[track_id] = True
                                self.entry_count += 1
                                print(f"New person entered! ID: {track_id}, Class: {cls}, Total unique entries: {self.entry_count}")
                            
                            # Draw bounding box
                            color = self.colors(int(track_id), True)
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(output_frame, f"ID: {track_id} (C{cls})", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Mark foot point
                            foot_x, foot_y = int((x1 + x2) / 2), y2
                            cv2.circle(output_frame, (foot_x, foot_y), 5, color, -1)
                            
                            # Mark region status
                            status = "IN" if is_in_region else "OUT"
                            cv2.putText(output_frame, status, (x1, y2+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    print(f"Error processing tracking results: {e}")
            else:
                if frame_count % 10 == 0:
                    print("No tracking IDs available in this frame")
        elif frame_count % 10 == 0:
            print("No detections in this frame")
        
        # Draw statistics
        output_frame = self.draw_statistics(output_frame)
        
        return output_frame
    
    def process_video(self):
        """Process the entire video"""
        frame_count = 0
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                print("Video frame is empty or processing is complete.")
                break
            
            # Update frame count
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}")
            
            # Process the frame
            output_frame = self.process_frame(frame, frame_count)
            
            # Display and write
            cv2.imshow("Region Persistent Counter", output_frame)
            self.video_writer.write(output_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Output saved to {self.output_path}")
        print(f"Total unique people entered the region: {self.entry_count}")
        
        # Print class statistics
        if self.class_counts:
            print("Class detection statistics:")
            for cls, count in self.class_counts.items():
                print(f"  Class {cls}: {count} detections")


# Example usage
if __name__ == "__main__":
    # Video path
    video_path = "/Users/manhcuongfx/Downloads/ultralytics-main/ultralytics/abc.mp4"
    
    # Region points
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    
    # Output path with .mp4 extension
    output_path = "region_counting.mp4"
    
    # Create and run the counter
    counter = RegionPersistentCounter(
        video_path=video_path,
        region_points=region_points,
        model_path="yolo11m.pt",
        output_path=output_path
    )
    
    # Process the video
    counter.process_video()