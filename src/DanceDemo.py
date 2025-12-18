import numpy as np
import cv2
import os
import pickle
import sys

from VideoSkeleton import VideoSkeleton, combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN
from GenGAN import GenGAN


class DanceDemo:
    """
    Class that runs a dance demo.
    The motion/posture from self.source is applied to the character defined by self.target using self.generator.
    """

    def __init__(self, filename_src, typeOfGen=2):
        # Target video (defines the style / appearance)
        self.target = VideoSkeleton("../data/taichi1.mp4")

        # Source video (defines the motion)
        self.source = VideoReader(filename_src)

        # Select generator
        if typeOfGen == 1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen == 2:         # VanillaNN, skeleton vector -> image
            print("Generator: GenVanillaNN (Skeleton Vector)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)
        elif typeOfGen == 3:         # VanillaNN, skeleton image -> image (U-Net)
            print("Generator: GenVanillaNN (Skeleton Image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)
        elif typeOfGen == 4:         # GAN
            print("Generator: GenGAN (Skeleton Image)")
            self.generator = GenGAN(self.target, loadFromFile=True)
        else:
            print("DanceDemo: invalid typeOfGen value!")
            self.generator = None

        self.typeOfGen = typeOfGen

    def draw(self):
        if self.generator is None:
            print("DanceDemo: No generator initialized.")
            return

        # Red error image if no skeleton is detected
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)

        total_frames = self.source.getTotalFrames()
        print(f"DanceDemo: total source frames = {total_frames}")

        ske = Skeleton()

        for i in range(total_frames):
            image_src = self.source.readFrame()
            if image_src is None:
                break

            # Process every 5th frame for smoother demo
            if i % 5 == 0:
                # Resize source frame to a reasonable width (like VideoSkeleton does)
                h, w = image_src.shape[:2]
                new_width = 500
                new_height = int(h * new_width / w)
                image_src_resized = cv2.resize(image_src, (new_width, new_height))

                # Extract skeleton directly from the resized source frame
                isSke = ske.fromImage(image_src_resized)

                if isSke:
                    # Draw skeleton on the source image (for visualization)
                    ske.draw(image_src_resized)

                    # Generate target image from skeleton
                    image_tgt = self.generator.generate(ske)

                    # Display size
                    display_size = (256, 256)
                    image_tgt = cv2.resize(image_tgt, display_size)
                else:
                    image_src_resized = cv2.resize(image_src, (256, 256))
                    image_tgt = image_err

                # Ensure source image matches target display size for combining
                image_src_disp = cv2.resize(image_src_resized, image_tgt.shape[1::-1])

                image_combined = combineTwoImages(image_src_disp, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))

                cv2.imshow('DanceDemo', image_combined)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    # Skip ahead in the source video
                    self.source.readNFrames(100)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # NEAREST = 1
    # VANILLA_NN_SKE = 2
    # VANILLA_NN_IMAGE = 3
    # GAN = 4
    GEN_TYPE = 4

    ddemo = DanceDemo("../data/taichi2.mp4", GEN_TYPE)
    # ddemo = DanceDemo("../data/tricking.mp4", GEN_TYPE)
    ddemo.draw()
