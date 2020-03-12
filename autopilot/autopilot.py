"""
.. module:: autopilot
   :synopsis: Main routine for autopilot package
.. moduleauthor:: Adam Moss <adam.moss@nottingham.ac.uk>
"""

import threading
import tensorflow as tf
import cv2

class AutoPilot:

    def __init__(self, capture, front_wheels, back_wheels, camera_control,
                 debug=False, test_mode=False):

        # Try getting camera from already running capture object, otherwise get a new CV2 video capture object
        try:
            self.camera = capture.camera
        except:
            import cv2
            self.camera = cv2.VideoCapture(0)

        # These are picar controls
        self.front_wheels = front_wheels
        self.back_wheels = back_wheels
        self.camera_control = camera_control

        self.debug = True #debug
        self.test_mode = test_mode

        # Thread variables
        self._started = False
        self._terminate = False
        self._thread = None

    def start(self):
        """
        Starts autopilot in separate thread
        :return:
        """
        if self._started:
            print('[!] Self driving has already been started')
            return None
        self._started = True
        self._terminate = False
        self._thread = threading.Thread(target=self._drive, args=())
        self._thread.start()

    def stop(self):
        """
        Stops autopilot
        :return:
        """
        self._started = False
        self._terminate = True
        if self._thread is not None:
            self._thread.join()

    def _drive(self):
        """
        Drive routine for autopilot. Processes frame from camera
        :return:
        """
        while not self._terminate:
            ret, frame = self.camera.read()

            # !! Use machine learning to determine angle and speed (if necessary - you may decide to use fixed speed) !!

            interpreter = tf.lite.Interpreter("/home/pi/autopilot/models/converted_model.tflite")
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Test model on random input data.
            input_shape = input_details[0]['shape']
            
            new_data_shape = (int(input_shape[1]),int(input_shape[2]))
            #need to reshape the frame into input data
            input_data = cv2.resize(frame, new_data_shape)
            #
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            angle = output_data[0]
            speed = output_data[1]

            # !! End of machine learning !!

            angle = int(angle)
            speed = int(speed)
            print('Speed: %d, angle: %d ' % (angle, speed))
            if self.debug:
                print('Speed: %d, angle: %d ' % (angle, speed))

            if not self.test_mode:

                # Do not allow angle or speed to go out of range
                angle = max(min(angle, self.front_wheels._max_angle), self.front_wheels._min_angle)
                speed = max(min(speed, 100), -100)

                # Set picar angle and speed
                self.front_wheels.turn(angle)
                self.back_wheels.forward()
                self.back_wheels.speed = speed

        self.back_wheels.speed = 0
