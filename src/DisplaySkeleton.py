import cv2
import sys
from VideoSkeleton import VideoSkeleton

def display_skeletons(video_file, image_size=512, delay_ms=100):
    """Display rendered skeleton images from a video file."""
    print(f"Loading video: {video_file}")
    targetVideoSke = VideoSkeleton(video_file)
    
    print(f"Total frames: {targetVideoSke.skeCount()}")
    print(f"Skeleton dimension: {targetVideoSke.ske[0].__array__(reduced=True).shape if targetVideoSke.skeCount() > 0 else 'N/A'}")
    
    paused = False
    for i in range(targetVideoSke.skeCount()):
        ske = targetVideoSke.ske[i]
        
        # Render skeleton to image using SkeToImageTransform
        import numpy as np
        from GenVanillaNN import SkeToImageTransform
        
        transform = SkeToImageTransform(image_size)
        ske_img = transform(ske)
        
        # Convert RGB back to BGR for OpenCV display
        ske_img_bgr = cv2.cvtColor(ske_img, cv2.COLOR_RGB2BGR)
        
        # Resize for display
        display_size = (512, 512)
        ske_img_display = cv2.resize(ske_img_bgr, display_size)
        
        # Add frame counter
        cv2.putText(ske_img_display, f"Frame {i+1}/{targetVideoSke.skeCount()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Skeleton Visualization', ske_img_display)
        
        if paused:
            print(f"Paused at frame {i+1}")
            while True:
                key = cv2.waitKey(0)
                if key == ord(' '):
                    paused = False
                    break
                elif key == 27 or key == ord('q'):  # ESC or Q
                    cv2.destroyAllWindows()
                    return
        else:
            key = cv2.waitKey(delay_ms)
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord(' '):  # SPACE to pause
                paused = True
    
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = "../data/taichi1.mp4"
    
    image_size = 512
    delay_ms = 100  # 100ms between frames
    
    display_skeletons(video_file, image_size, delay_ms)
