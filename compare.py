import cv2
import easyocr
import csv

# Load the pre-trained cascade classifier for detecting license plates
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Open the CSV file with the license plate numbers
with open('numbers.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    # Skip the header row
    next(csv_reader)
    # Create a set to store the license plate numbers
    numbers_set = set(row[0] for row in csv_reader)

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

# Check if the capture device is open
if not cap.isOpened():
    print("Could not open video capture device")
    exit()

# Loop until a match is found or the user presses 'q' to quit
while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # If the frame was not captured successfully, exit the loop
    if not ret:
        break

    # Convert the color image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through each license plate detected
    for (x, y, w, h) in plates:
        # Crop the license plate from the image
        plate_img = frame[y:y+h, x:x+w]

        # Use EasyOCR to read the license plate number
        results = reader.readtext(plate_img)

        # If EasyOCR successfully reads the license plate number, check if it is in the set of numbers
        if len(results) > 0:
            number = results[0][1]
            if number in numbers_set:
                print(f"{number} is present in database")
                # Stop displaying the video stream
                cap.release()
                cv2.destroyAllWindows()
                # Take a picture of the license plate and save it
                cv2.imwrite(f"{number}.png", plate_img)
                exit()
            else:
                print(f"{number} is not present in database")
                # Capture the next frame
                break

        # Draw a rectangle around the license plate in the original color image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with the license plate detection and recognition results
    cv2.imshow("License Plate Detection and Recognition", frame)

    # Wait for a key press and check if 'q' was pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
