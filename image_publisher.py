#!/usr/bin/env python

#import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

from termcolor import colored
import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "192.168.68.73"  # Change to your MQTT broker address
MQTT_PORT = 1883  # Change to your MQTT broker port
MQTT_TOPIC = "patient/fall"  # Change to your desired topic

# Initialize MQTT Client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

last_sent_time = 0

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" % port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " % (backendName, h, w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args) <= 1 and len(port_ids) > 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0 # limit to ~10 fps here
            cv2.namedWindow('edgeimpulse', cv2.WINDOW_NORMAL)

            expansion_factor = 0.2  # Expand bounding boxes by 20%

            for res, img in runner.classifier(videoCaptureDeviceId):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                if "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        x, y, width, height = bb['x'], bb['y'], bb['width'], bb['height']
                        # Calculate expansion
                        expand_width = width * expansion_factor
                        expand_height = height * expansion_factor
                        # Apply expansion to the original coordinates
                        x_expanded = max(0, x - expand_width / 2)
                        y_expanded = max(0, y - expand_height / 2)
                        width_expanded = width + expand_width
                        height_expanded = height + expand_height
                        # Ensure expanded box does not exceed image dimensions
                        img_height, img_width = img.shape[:2]
                        width_expanded = min(img_width - x_expanded, width_expanded)
                        height_expanded = min(img_height - y_expanded, height_expanded)
                        # Draw the expanded bounding box
                        img = cv2.rectangle(img, (int(x_expanded), int(y_expanded)), (int(x_expanded + width_expanded), int(y_expanded + height_expanded)), (255, 0, 0), 1)
                        Label = bb['label']
                        confidence = bb['value']
                        print(f"Bounding Box: (x={x}, y={y}), Width={width}, Height={height}")
                        print(f"Label: {Label}, Confidence: {confidence}")
                        
                        if Label == "Standing":
                            new_label = "Stationary"
                        elif Label == "Fall":
                            new_label = "Disturbance Detected"
                        else:
                            new_label = "Unknown"
                            
                        # Publish all data to MQTT subscriber
                        payload = f"Ward 1, {time.strftime('%Y-%m-%d %H:%M:%S')}, Label: {new_label}, Confidence: {confidence}\n"
                        mqtt_client.publish(MQTT_TOPIC, payload=payload, qos=0, retain=False)
                        time.sleep(0.0)
                            

                if (show_camera):
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
