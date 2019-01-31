#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jon Binney
# '''
# Functions for working with PointCloud2.
# '''

# __docformat__ = "restructuredtext en"
import os
import random
import rospy
from gaips_msgs.srv import speak_req, speak_reqResponse


# import rospy, cv, cv2, sys, os, numpy, math, time, random, copy
# import roslib; roslib.load_manifest('sensor_msgs')
# import numpy as np
# from sensor_msgs.msg import PointCloud2, PointField
# from tf import transformations
# from geometry_msgs.msg import Transform, Vector3, Quaternion


# from sensor_msgs.msg import Image, PointCloud2
# from cv_bridge import CvBridge, CvBridgeError
# from cv_bridge import CvBridge, CvBridgeError
# from gaips_msgs.msg import Face, Faces




class VoiceManager:
    def __init__(self):

        # Espeak in-line command parameter settings
        # self.espeak_str = " espeak -s 175 -p 90 -g 10 "
        # self.espeak_str = " espeak -v pt-pt -p 90 -a 75 -s 190 "
        self.espeak_str = " espeak -v mb-en1 -p 85 -a 185 -s 145 -g 2 "

        # Service setup
        self.t = rospy.Service('/bea/speak', speak_req, self.speakEmotion)

        # Sentence to be said
        self.sentence = None

        self.something_to_say = False
        self.string = ""

    def speakEmotion(self, speak_req):

        # Depending on the emotion desired, choose a self.sentence adequate
        if speak_req.sentence == speak_req.GREETING:
            self.chooseGreeting()
        elif speak_req.sentence == speak_req.GOODBYE:
            self.chooseGoodbye()
        elif speak_req.sentence == speak_req.HAPPINESS:
            self.chooseHappiness()
        elif speak_req.sentence == speak_req.EXCITATION:
            self.chooseExcitation()
        elif speak_req.sentence == speak_req.SADNESS:
            self.chooseSadness()
        elif speak_req.sentence == speak_req.SURPRISE:
            self.chooseSurprise()
        elif speak_req.sentence == speak_req.SLEEPY:
            self.chooseSleepy()
        elif speak_req.sentence == speak_req.NERVOUS:
            self.chooseNervous()
        else:
            # No valid emotion, don't say nothing and issue warning
            rospy.logerr("Not a valid emotion to speak.")
            response = speak_reqResponse()
            response.result = False
            return response

        self.something_to_say = True

        # Request response
        response = speak_reqResponse()
        response.result = True
        return response

        # Speak the chosen sentence
        #self.speakString()



    def chooseGreeting(self):

        sentences = ["Olá, o meu nome é ",
                     "Olá, eu sou "]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseGoodbye(self):

        sentences = ["See you next time!",
                     "I loved you guys, 'till next time.",
                     "It's all for now, see you later."]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseHappiness(self):

        sentences = ["Boa",
                     "Fiche"]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseExcitation(self):

        sentences = ["Sim! ",
                     "Consegui! ",
                     "Fantástico! "]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseSadness(self):

        sentences = ["Oh não",
                     "Óo",
                     "Bolas",
                     "Fogo"]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseSurprise(self):

        sentences = ["Uau! ",
                     "Olá! "]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseSleepy(self):

        sentences = ["Estou com sono",
                     "Que sono "]

        self.string = sentences[random.randrange(len(sentences))]

    def chooseNervous(self):

        sentences = ["É difícil",
                     "Puxa ",
                     "É desta"]

        self.string = sentences[random.randrange(len(sentences))]

    def choose_cup_filling_greeting(self, greeting_code):

        """
        greetings = {
            1: ["Olá, o meu nome é " + "\"" + "; espeak -v en-us -p 90 -g 15 -s 145 \"" + "Baxter" + "\"" +
                "; espeak -v pt-pt -p 90 -g 15 -s 145 \" e hoje vou servir-vos água.",
                "Olá, sou u " + "\"" + "; espeak -v en-us -p 90 -g 15 -s 145 \"" + "Baxter" + "\"" +
                "; espeak -v pt-pt -p 90 -g 15 -s 145 \" e vou ser o vosso bartendarr hoje"],
            2: ["Ainda com sede? Eu sei... Tem estado calor. Eu encho-vos esses copos.",
                "De volta!? Estou a ver que só um copo não vos chegou, eu encho-vos outro."],
            3: ["Querem mais!? Vocês gostaram mesmo de mim! Então vá dêem cá esses copos.",
                "Mais!? Onde põe vocês essa água? Vamos lá, não vos posso deixar com sede!"]
        }
        """

        greetings = {
            1: ["Olá, o meu nome é " + "\"" + "; espeak -v en-us -p 90 -g 15 -s 145 \"" + "Baxter" + "\"" +
                "; espeak -v pt-pt -p 90 -g 15 -s 145 \" e hoje vou servir-vos água.",
                "Olá, sou u " + "\"" + "; espeak -v en-us -p 90 -g 15 -s 145 \"" + "Baxter" + "\"" +
                "; espeak -v pt-pt -p 90 -g 15 -s 145 \" e vou ser o vosso bartendarr hoje"],
            2: ["Still thirsty?? I get it... it has been hot... I'll serve you another one.",
                "Again? I see one wasn't enough. I'll give you another one!"],
            3: ["Want more? You really liked me! Come here, I'll give you more.",
                "More? Where do you put all that water?? I'll pour you another one."]
        }

        self.string = greetings[greeting_code][random.randrange(len(greetings[greeting_code]))]

    def speakGreeting(self):

        command_str = self.espeak_str + "\"" + self.string + "\"" + "; espeak -v en-us -p 35 \"" + "Baxter" + "\""
        print(command_str)
        os.system(command_str)

    def speakString(self, string = ""):

        if not string:
            string = self.string

        # Issue a terminal command
        command_str = self.espeak_str + "\"" + string + "\""
        print(command_str)
        os.system(command_str)

        self.something_to_say = False

def main():

    rospy.init_node('Bea_voice_manager')

    try:

        vm = VoiceManager()

        rospy.loginfo("Bea Voice Manager Started")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if vm.something_to_say:
                vm.speakString()
            rate.sleep()
    except KeyboardInterrupt:
        rospy.logerr("Received Keyboard shutdown: Shutting down")
    except Exception as e:
        rospy.logerr(e)

if __name__ == '__main__':
    main()
