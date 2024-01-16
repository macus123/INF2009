import paho.mqtt.client as mqtt
import time

# MQTT settings
MQTT_BROKER = "masked"
MQTT_PORT = 1883
MQTT_TOPIC = "image/capture"

client = mqtt.Client("Publisher")
client.connect(MQTT_BROKER, MQTT_PORT)

try:
    while True:
        # Send a capture command
        client.publish(MQTT_TOPIC, "Capture")
        print("Capture command sent.")
        time.sleep(5)
except KeyboardInterrupt:
    print("Exiting...")

client.disconnect()
