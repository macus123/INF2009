import paho.mqtt.client as mqtt
import cv2

# MQTT settings
MQTT_BROKER = "masked"
MQTT_PORT = 1883
MQTT_TOPIC = "image/capture"

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def on_message(client, userdata, message):
    print(f"Received capture command on topic '{message.topic}'")

    # Capture one frame
    ret, frame = cap.read()
    if ret:
        # Save the captured image
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured and saved!")

        # Publish the image
        with open('captured_image.jpg', 'rb') as f:
            image_data = f.read()
        client.publish("image/publish", image_data)

client = mqtt.Client("Subscriber")
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)
client.subscribe(MQTT_TOPIC)

# Start the loop
client.loop_start()

# Keep the script running
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    client.loop_stop()
