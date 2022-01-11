import cv2
import numpy as np
from src.hole_filling import HF
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
  parser = argparse.ArgumentParser(description='depthInpainting')

  parser.add_argument('--input', default='', type=str)
  parser.add_argument('--output_dir', default='./visualization', type=str)
  parser.add_argument('--threshold', default=48, type=int)
  parser.add_argument('--hole_dilate_iter', default=0, type=int)
  parser.add_argument('--boundary_dilate_iter', default=1, type=int)
  parser.add_argument('--dilate_kernel_size', default=3, type=int)

  args = parser.parse_args()
  return args

def cluster(img):
  labels = cv2.connectedComponents(img)[1]
  label_hue = np.uint8(179*labels/np.max(labels))
  blank_ch = 255*np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
  labeled_img[label_hue==0] = 0

  return labeled_img

def main(args):

  # params
  original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

  os.makedirs(args.output_dir, exist_ok=True)

  cv2.imwrite(f'{args.output_dir}/original.png', original)

  hf = HF(original, threshold=args.threshold, hole_dilate_iter=args.hole_dilate_iter, boundary_dilate_iter=args.boundary_dilate_iter, dilate_kernel_size=(args.dilate_kernel_size, args.dilate_kernel_size))

  holes = hf.get_holes()

  cv2.imwrite(f'{args.output_dir}/holes.png', holes)

  holes_dilated = hf.dilate(holes, args.hole_dilate_iter)

  cv2.imwrite(f'{args.output_dir}/holes_dilated.png', holes_dilated)

  boundries = hf.dilate(holes_dilated, args.boundary_dilate_iter) - holes_dilated

  cv2.imwrite(f'{args.output_dir}/boundries.png', boundries)

  holes_cluster = cluster(holes_dilated)

  cv2.imwrite(f'{args.output_dir}/holes_cluster.png', holes_cluster)

  boundries_cluster = cluster(boundries)

  cv2.imwrite(f'{args.output_dir}/boundries_cluster.png', boundries_cluster)

  # create histogram
  num_label, boundries_label = cv2.connectedComponents(boundries)
  inpainting_candidates = original[boundries_label == np.random.randint(num_label)]
  plt.figure(figsize=(16, 9), dpi=300)
  plt.hist(inpainting_candidates, bins=np.arange(args.threshold, 257, 1), density=True)
  plt.xlabel("depth")
  plt.ylabel("Probability")
  plt.title("The Boundary Depth of One Hole")
  plt.savefig(f"{args.output_dir}/historgram.png", bbox_inches='tight')

  repaired = hf.repair()
  cv2.imwrite(f'{args.output_dir}/repaired.png', repaired)

if __name__ == '__main__':
    args = parse_args()
    main(args)