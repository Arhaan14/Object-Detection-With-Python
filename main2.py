import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "C:/full/path/to/yolov3.cfg")

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image
img = cv2.imread("image.jpg")

# Get image dimensions
height, width, _ = img.shape

# Create blob from the image
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set input for the model
net.setInput(blob)

# Forward pass through the model
outs = net.forward(net.getUnconnectedOutLayersNames())

# Initialize lists to store detected objects' details
class_ids = []
confidences = []
boxes = []

# Loop over each output layer
for out in outs:
    # Loop over each detection
    for detection in out:
        # Extract the class ID and confidence score
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Calculate the center and width/height of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            # Store the details of the detected object
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop over the indices and draw the bounding boxes with class labels
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = colors[class_ids[i]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with the detected objects
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
