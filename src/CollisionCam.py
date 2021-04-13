# Dune Collision Camera
# written by EB Goldstein and SD Mohanty
# started 10/2020
# last revision 4/2021

# import what we need
import pyb, sensor, image, time, os, tf, random

#setup LEDs and set into known off state
red_led = pyb.LED(1)
green_led = pyb.LED(2)
red_led.off()
green_led.off()

#red light during setup
red_led.on()

# get sensor set up
sensor.reset()                         # Reset & initialize sensor
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB
sensor.set_framesize((sensor.QVGA))      # Set frame size to QVGA (320x240)
sensor.set_windowing((240,240))       # Set window to 240x240
sensor.skip_frames(time=2000)          # Let the camera adjust.

#Load the TFlite model and the labels
net = tf.load('/post_quantized_full_int.tflite', load_to_fb=True)
labels = ['collision', 'no collision']

#turn red off when model is loaded
red_led.off()

#MAIN LOOP

# loop needs to do a few things:
# x-take a picture
# x-record the picture on sd card
# x-do the inference
# x-record the inference in a db w/ rand as name
# x-blink LED for field debugging
# x-delay

while(True):

    #toggle LED for visual indication that script is running
    green_led.toggle()

    #get the image/take the picture
    img = sensor.snapshot()

    #Do the classification and get the object returned by the inference.
    TF_objs = net.classify(img)

    #The object has a output, which is a list of classifcation scores
    #for each of the output channels. this model only has 1 (Collision).
    #So now we extract that float value and print to the serial terminal.

    collision_score = TF_objs[0].output()[0]
    print("Collision = %f" % collision_score)

    #we don;t have an RTC attached now, so we save the images and the
    #collision scores with names according to a random bit stream.

    #generate random bits for stream of names
    rand_label = str(random.getrandbits(30))

    #save image on camera (bit as name)
    img.save("./imgs/" + rand_label + ".jpg")

    #save inference (bit as name, and score) to a file
    with open("./inference.txt", 'a') as file:
        file.write(rand_label + "," + str(collision_score) + "\n")

    #wait some number of milliseconds
    pyb.delay(1000)
