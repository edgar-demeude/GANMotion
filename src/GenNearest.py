import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """

        min_dist = float('inf')
        min_img = None
        for i in range (len(self.videoSkeletonTarget.ske)-1) :
            if ske.distance(self.videoSkeletonTarget.ske[i]) < min_dist :
                min_dist = ske.distance(self.videoSkeletonTarget.ske[i])
                min_img = self.videoSkeletonTarget.im[i]
        print("Distance min = ", min_dist)
        print("Image min = ", min_img)
        image = cv2.imread("../data/" + min_img)

        return image