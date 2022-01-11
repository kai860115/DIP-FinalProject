import numpy as np
import cv2

class HF:
    def __init__(self, original, threshold=24, hole_dilate_iter=0, boundary_dilate_iter=1, dilate_kernel_size=(3,3)):
      self.original = original
      self.threshold = threshold
      self.hole_dilate_iter = hole_dilate_iter
      self.boundary_dilate_iter = boundary_dilate_iter
      self.dilate_kernel_size = dilate_kernel_size

    def get_original(self):
      return self.original

    def get_holes(self,img, threshold):
      return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    def dilate(self, src, itr):
      kernel = np.ones(self.dilate_kernel_size, np.uint8)
      return cv2.dilate(src, kernel, iterations = itr)

    def get_cluster(self, img):
      label = cv2.connectedComponents(img)[1]

      return label

    def calc_filling_color(self, img, filter):
      candidates = img * filter
      pr, edge = np.histogram(candidates, bins=np.arange(self.threshold, 257, 1), density=True)
      # return edge[np.argmax(pr)]
      return np.sum(pr * edge[:-1])

    def repair(self):
      repaired = np.copy(self.original)

      holes = self.get_holes(self.original, self.threshold)
      holes = self.dilate(holes, self.hole_dilate_iter)
      num_label, holes_label = cv2.connectedComponents(holes)


      for i in range(1, num_label):

        hole = np.zeros(self.original.shape)
        hole[holes_label == i] = 1

        dilated = self.dilate(hole, self.boundary_dilate_iter)
        boundary = dilated - hole

        depth = self.calc_filling_color(self.original, boundary)
        repaired[holes_label == i] = depth

      return repaired