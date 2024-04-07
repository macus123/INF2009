import os
import sys, getopt
import signal
import paho.mqtt.client as mqtt  # Import the MQTT client
import time
from edge_impulse_linux.audio import AudioImpulseRunner

runner = None

# MQTT Configuration
MQTT_BROKER = "192.168.68.73"  # Change to your MQTT broker address
MQTT_PORT = 1883  # Change to your MQTT broker port
MQTT_TOPIC = "patient/distress"  # Change to your desired topic

# Initialize MQTT Client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False
    
def now():
    return round(time.time())

def signal_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    mqtt_client.disconnect()  # Disconnect from MQTT broker
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def help():
    print('python classify.py <path_to_model.eim> <audio_device_ID, optional>' )

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

    with AudioImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')

            #Let the library choose an audio interface suitable for this model, or pass device ID parameter to manually select a specific audio interface
            selected_device_id = None
            if len(args) >= 2:
                selected_device_id=int(args[1])
                print("Device ID "+ str(selected_device_id) + " has been provided as an argument.")

            for res, audio in runner.classifier(device_id=selected_device_id):
                            distress_detected = False  # Flag to track distress detection
                            
                            for label in labels:
                                score = res['result']['classification'][label]
                                print('%s: %.2f\t' % (label, score), end='')
                                
                                # Check if label '1' has a score above 0.6
                                if label == '1' and score > 0.6:
                                    distress_detected = True

                            print('', flush=True)
                            
                            # Collect additional data
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                            # Determine the patient's state
                            patient_state = "Distressed" if distress_detected else "Calm"
                            last_classification_value = res['result']['classification']['1']


                            print(f'Result ({res["timing"]["dsp"] + res["timing"]["classification"]} ms.) - Patient State: {patient_state}, Prediction: {last_classification_value}', end='')

                            # Publish the message for every result
                            mqtt_client.publish(MQTT_TOPIC, payload=f"Ward 1, {timestamp}, Patient State: {patient_state}, Prediction: {last_classification_value}\n", qos=0, retain=False)
                    


        finally:
            if (runner):
                runner.stop()

if __name__ == '__main__':
    main(sys.argv[1:])
