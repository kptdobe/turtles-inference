import os
import torch
from torchvision import transforms
import numpy as np
import time
import re
from pathlib import Path
from scipy.stats import wasserstein_distance  # For Wasserstein distance calculation
from itertools import compress
import glob
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin

from lightglue import LightGlue, SuperPoint, SIFT  # Import your LightGlue components
from lightglue.utils import load_image  # Import your image loading utility

torch.set_grad_enabled(False)

# Use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
print("Using device:", device)

path_root = "./models/"
# path_root = "/Users/acapt/Google Drive/AI/AI for Turtles Shared
# lightglue_path = path_root + "/05. Subgroup folders/Feature extraction/notebooks/LightGlue_Implementations"
# fpath_base = lightglue_path + "/WassersteinDistributions/"
# fpath_base_W = "/content/drive/.shortcut-targets-by-id/14M23_ygiJ6LB0JpUgESZxLyWOQQjp60h/LightGlue_Implementations/WassersteinDistributions/"
fpath_base = path_root
fpath_base_W = path_root
# save_path = '/content/drive/.shortcut-targets-by-id/14M23_ygiJ6LB0JpUgESZxLyWOQQjp60h/LightGlue_Implementations/'
# save_path = lightglue_path + "/"
save_path = path_root

# Path to the base directory of images
# base_image_path = path_root + "/05. Subgroup folders/Face detection /Data/Extracted Turtle Faces cropped"
# Path to the uploaded image
# input_image_path = "/content/drive/MyDrive/22-119 R.jpg"  # Replace with the path of your uploaded image
# input_image_path = "/Users/acapt/Downloads/drive-download-20231113T134801Z-001/22-111 L.JPG"

def label_from_path(path):
  # match = re.search(r'(\d{2})-(\d+) ([LR]).JPG', path)
  # year, num, side = match.groups()

  # return f"{year}-{num} {side}"

  # Extract year, number, and side (if available) from the filename
  match = re.search(r'(\d{2})-(\d+)', path)

  if match:
      year, number = match.groups()

      # Check for capital 'L' or 'R' in the filename
      side_match = re.search(r'[LR]', path)

      if side_match:
          side = side_match.group()
      else:
          # If capital 'L' or 'R' not found, check for lowercase 'l' or 'r'
          side_match = re.search(r'[lr]', path)
          side = side_match.group().upper() if side_match else ''

      # Create a new filename based on the specified format
      return f"{year}-{number} {side}"
  
def feature_dict_to_device(f_dict, device):
  for key in f_dict:
    temp_dict = {}

    for f_key in f_dict[key]:
      temp_dict[f_key] = f_dict[key][f_key].to(device)
    f_dict[key] = temp_dict

  return f_dict

def divide_list(original_list, N):
    num_sublists = len(original_list) // N
    remainder = len(original_list) % N

    sublists = [original_list[i * N:(i + 1) * N] for i in range(num_sublists)]

    if remainder > 0:
        sublists.append(original_list[-remainder:])

    return sublists

def divide_list(original_list, N):
    num_sublists = len(original_list) // N
    remainder = len(original_list) % N

    sublists = [original_list[i * N:(i + 1) * N] for i in range(num_sublists)]

    if remainder > 0:
        sublists.append(original_list[-remainder:])

    return sublists

def import_distributions(fpath_base,method='superpoint'):
  if method == 'disk':
    f0 = 'x0_disk.npy'
    f1 = 'x1_disk.npy'
  elif method == 'superpoint':
    f0 = 'x0_superpoint.npy'
    f1 = 'x1_superpoint.npy'
  elif method == 'aliked':
    f0 = 'x0_aliked.npy'
    f1 = 'x1_aliked.npy'
  elif method == 'sift':
    f0 = 'x0_sift.npy'
    f1 = 'x1_sift.npy'

  fp0 = fpath_base + f0
  fp1 = fpath_base + f1

  x0 = np.load(fp0)
  x1 = np.load(fp1)
  return x0, x1

def batch_by_dict(feature_dict, labels, max_batch_size=32):
  kpt_len = np.array([len(feature['keypoints'][0]) for feature in feature_dict.values()])

  feature_list = feature_dict.values()
  label_list = []
  batch_list = []
  for n_kpt in set(kpt_len):
    sub_feat = list(compress(feature_list, kpt_len == n_kpt))
    sub_labels = list(compress(labels, kpt_len == n_kpt))

    # if size of subset > batch size, split into smaller batches
    bs_sub_feat = divide_list(sub_feat, max_batch_size)
    bs_sub_labels = divide_list(sub_labels, max_batch_size)

    for bs_feat, bs_labels in zip(bs_sub_feat, bs_sub_labels):
      new_dict = {}
      for key in bs_feat[0].keys():
        new_dict[key] = []

      for feat in bs_feat:
        for key in feat.keys():
          new_dict[key].append(feat[key])

      for key in new_dict:
        new_dict[key] = torch.concat(new_dict[key])

      batch_list.append(new_dict)
      label_list += bs_labels

  return batch_list, label_list

def matching_score(match_out, method="score10", x0=[], batch_index=0):
  '''
  method - the method of extraction
    'wasserstein_h0': The scipy function wasserstein_distance
    'wasserstein_h1': The scipy function wasserstein_distance, but ammended to be 1-wasserstein
  x0 - the distribution that we are comparing to
  '''
  match_scores = match_out["scores"][batch_index]
  im0_scores = match_out["matching_scores0"][batch_index]

  # top 10 matched keypoint scores summed
  if method == "score10":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 10)].sum()

  # top 25 matched keypoint scores summed
  elif method == "score25":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 25)].sum()

  # top 50 matched keypoint scores summed
  elif method == "score50":
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 50)].sum()

  # ------------
  # add Wasserstein_h0 here
  elif method == "wassersteinH0":
    return wasserstein_distance(x0[0], im0_scores.to("cpu").detach().numpy()), wasserstein_distance(x0[1], im0_scores.to("cpu").detach().numpy())

  # add Wasserstein_h1 here - this compares to a matched distribution (i.e. small wasserstein = match)
  elif method == "wassersteinH1":
    return 1 - wasserstein_distance(x0, im0_scores.to("cpu"))
  # ------------

  else:
    print("Unviable method argument. Defaulting to total number of matches.")
    return match_scores.sort(descending=True)[0][:min(len(match_scores), 10)].sum()


def predict(score, labels, distH0, topN=5, metric='Highest'):
  '''
  Returns the top N predicted turtle IDs based on an array of scores

  Args:
    score - array of scores for current image against whole reference set
    labels - list of turtle IDs corresponding to each score
    topN - top N predictions to return
    metric - what metric you are using (most use the largest score, apart from wassersteinH1)
  '''
  score, labels = np.array(score), np.array(labels)
  indices = np.argsort(score) #  default asc
  if metric == 'wassersteinH1':
    score_dict = {label: s for label, s in zip(labels[indices], score[indices])} # only minimum scores for each label
    reverse = False # ascending order

  else:
    score_dict = {label: s for label, s in zip(labels[indices], zip(score[indices], np.array(distH0)[indices]))} # only maximum scores for each label
    reverse = True # descending order

  sorted_items = sorted(score_dict.items(), key=lambda x: x[1][0], reverse=reverse) # sort by score
  topN_items = sorted_items[:topN]

  return [item[0] for item in topN_items], [item[1] for item in topN_items]

def single_im_inference(image, face_side, matcher, extractor_obj, config, topN=5):

  start = time.time()
  print("Loading train_dict_R", f"{save_path}trainR_{config['extractor']}_{config['n_kpts']}.pth")
  train_dict_R = torch.load(f"{save_path}trainR_{config['extractor']}_{config['n_kpts']}.pth")
  print("train_dict_R loaded in ", time.time() - start, "seconds")

  start = time.time()
  print("Loading train_dict_L", f"{save_path}trainL_{config['extractor']}_{config['n_kpts']}.pth")
  train_dict_L = torch.load(f"{save_path}trainL_{config['extractor']}_{config['n_kpts']}.pth")
  print("train_dict_L loaded in ", time.time() - start, "seconds")

  print("feature_dict_to_device")
  # to device
  train_dict_L = feature_dict_to_device(train_dict_L, device)
  train_dict_R = feature_dict_to_device(train_dict_R, device)

  print("feature_dict_to_device done")

  print("Loading turtles_in_trainL and turtles_in_trainR")
  # reference set labels
  turtles_in_trainL = [label_from_path(im_path) for im_path in train_dict_L]
  turtles_in_trainR = [label_from_path(im_path) for im_path in train_dict_R]

  print("turtles_in_trainL and turtles_in_trainR loaded")

  # define batches and sort labels accordingly
  batch_list_L, turtles_in_trainL = batch_by_dict(train_dict_L, turtles_in_trainL, max_batch_size=32)
  batch_list_R, turtles_in_trainR = batch_by_dict(train_dict_R, turtles_in_trainR, max_batch_size=32)

  print("batch_list_ * loaded")

  # import reference distributions
  x0, x1 = import_distributions(fpath_base_W, method=config['extractor'])

  print("import_distributions loaded")

  # load image and extract features
  t_s = time.time()
  # image = load_image(new_impath).to(device)
  print(f"Extractor type: {type(extractor_obj)}")
  print(f"Extractor value: {extractor_obj}")
  nb_feat0 = extractor_obj.extract(image)
  t_e = time.time()
  time_load_extract = t_e - t_s

  # define scoring function
  metric = config["metric"]
  if metric == 'wassersteinH0':
    scoring_function = lambda out, i: matching_score(out, method=metric, x0=(x0, x1), batch_index=i)
  elif metric == 'wassersteinH1':
    scoring_function = lambda out, i: matching_score(out, method=metric, x0=x1, batch_index=i)
  else:
    scoring_function = lambda out, i: matching_score(out, method=metric, batch_index=i)

  # only search left/right reference set
  batch_list, turtles_in_train =  (batch_list_L, turtles_in_trainL) if face_side == "L" else (batch_list_R, turtles_in_trainR)

  scores = set()
  distsH1 = set()
  nr_matches = set()
  # match
  start = time.time();
  for batch in batch_list:
    s = time.time()
    print("Matching")
    batch_size = batch['keypoints'].shape[0]

    # reformat feat0 to fit batch
    feat0 = {}
    for key in nb_feat0:
      feat0[key] = torch.concat([nb_feat0[key]] * batch_size)

    matches = matcher({"image0": feat0, "image1": batch})

    for i in range(batch_size):
      # obtain scores
      # ss = time.time()
      # print("Scoring...")
      score, distH1 = scoring_function(matches, i)
      # print("Scored in ", time.time() - ss, "seconds")
      scores.add(score.item())
      distsH1.add(distH1.item())

      # nr of matched keypoints
      # nr_matches.add(len(matches["matches"][i]))

    print("Matched in ", time.time() - s, "seconds")

  print("Looped in ", time.time() - start, "seconds")
  scores = list(scores)
  distsH1 = list(distsH1)
  nr_matches = list(nr_matches)

  # predict
  # new turtle condition
  no_match_thresh = config["no_match_thresh"]
  is_new = False
  #Different metrics have different ways to detect novelty - starting by adding the wassersteinH0, more to come
  if metric == 'wassersteinH0':
    mxScore = np.max(scores)
    if mxScore < no_match_thresh:
      is_new = True #If highest score is below no_match_thresh it is too close to the null distribution
  else:
    if all(nr_m <= no_match_thresh for nr_m in nr_matches):
      is_new = True

  # top 5 matches
  print("Predicting...")
  predicted_label, predicted_scores = predict(scores, turtles_in_train, distsH1, metric=metric)
  print("Predicted")
  return predicted_label, predicted_scores, is_new, time_load_extract, time.time() - start

def find_image_path(base_path, image_name):
    # Iterate through each year's subfolder
    for year_folder in os.listdir(base_path):
        year_path = os.path.join(base_path, year_folder)
        if os.path.isdir(year_path):
            # Search for the image in this year's folder
            search_pattern = os.path.join(year_path, f"{image_name}.png")
            found_images = glob.glob(search_pattern)
            if found_images:
                return found_images[0]  # Return the first match
    return None  # Return None if the image is not found

def display_images(input_image_path, matched_image_info, base_path):
    plt.figure(figsize=(20, 10))

    # Display input image
    plt.subplot(1, len(matched_image_info) + 1, 1)
    input_image = Image.open(input_image_path)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')

    # Display top N matches
    for i, (title, confidence) in enumerate(matched_image_info, start=2):
        image_path = find_image_path(base_path, title)
        if image_path:
            image = Image.open(image_path)
            plt.subplot(1, len(matched_image_info) + 1, i)
            plt.imshow(image)
            plt.title(f"{title}\nConf: {confidence[0]:.2f}")
            plt.axis('off')
        else:
            print(f"Image not found: {title}")

    plt.show()


# Base Inference Class
class BaseInference:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def preprocess_image(self, image_path):
        # Implement image loading and preprocessing
        raise NotImplementedError("This method should be overridden.")

    def extract_features(self, image):
        # Implement feature extraction
        raise NotImplementedError("This method should be overridden.")

    def match_features(self, features):
        # Implement feature matching
        raise NotImplementedError("This method should be overridden.")

    def get_top_n_matches(self, matches, top_n):
        # Implement logic to return top N matches
        raise NotImplementedError("This method should be overridden.")

    def infer(self, image_path, top_n):
        # General method for inference
        image = self.preprocess_image(image_path)
        features = self.extract_features(image)
        matches = self.match_features(features)
        return self.get_top_n_matches(matches, top_n)

class LightGlueInference(BaseInference):
    def __init__(self, model, extractor, device):
        super().__init__(model, device)
        self.extractor = extractor

    def preprocess_image(self, image_path):
        # Load the image using the LightGlue utility function
        image = load_image(image_path)

        # Convert the image to the appropriate device (CPU or GPU)
        image = image.to(self.device)

        return image

    def extract_features(self, image):
        # Extract features using the LightGlue model
        features = self.extractor.extract(image)
        return features

    def match_features(self, feat0, batch_list, config):
     scores = []
     distsH1 = []
     nr_matches = []

     # Define the scoring function based on the config
     if config["metric"] == 'wassersteinH0':
        scoring_function = lambda out, i: matching_score(out, method=config["metric"], x0=config['x0'], batch_index=i)
     elif config["metric"] == 'wassersteinH1':
        scoring_function = lambda out, i: matching_score(out, method=config["metric"], x0=config['x1'], batch_index=i)
     else:
        scoring_function = lambda out, i: matching_score(out, method=config["metric"], batch_index=i)

     for batch in batch_list:
        batch_size = batch['keypoints'].shape[0]

        # Reformat feat0 to fit batch
        feat0_batch = {key: torch.concat([feat0[key]] * batch_size) for key in feat0}

        # Perform matching using LightGlue model
        matches = self.model({"image0": feat0_batch, "image1": batch})

        for i in range(batch_size):
            # Obtain scores using the scoring function
            score, distH1 = scoring_function(matches, i)
            scores.append(score)
            distsH1.append(distH1)

            # Number of matched keypoints
            nr_matches.append(len(matches["matches"][i]))

     return scores, distsH1, nr_matches


    def get_top_n_matches(self, scores, labels, distsH1, topN=5, metric='Highest'):
     score, labels = np.array(scores), np.array(labels)
     indices = np.argsort(score)  # Default ascending

     # Sorting logic based on metric
     if metric == 'wassersteinH1':
        score_dict = {label: s for label, s in zip(labels[indices], score[indices])}
        reverse = False  # Ascending order
     else:
        score_dict = {label: s for label, s in zip(labels[indices], zip(score[indices], np.array(distsH1)[indices]))}
        reverse = True  # Descending order

     sorted_items = sorted(score_dict.items(), key=lambda x: x[1][0], reverse=reverse)
     topN_items = sorted_items[:topN]

     return [item[0] for item in topN_items], [item[1] for item in topN_items]

    def infer(self, image, config, topN=5):
     # Use the single_im_inference function with the actual extractor object
     return single_im_inference(image, config['face_side'], self.model, self.extractor, config, topN)


# Factory Function
def inference_factory(algorithm, model, device):
    if algorithm == 'lightglue':
        return LightGlueInference(model, device)
    elif algorithm == 'another_algorithm':
        return
    else:
        raise ValueError(f"Algorithm {algorithm} not supported")

def load_lightglue_model(config):
    # Extract the feature extractor type from the config
    extractor_type = config["extractor"]

    # Initialize the LightGlue model with the string identifier of the extractor
    model = LightGlue(features=extractor_type, n_layers=config["n_layers"],
                      width_confidence=-1, depth_confidence=-1).eval().to(device)

    # Create the feature extractor object for later use
    if extractor_type == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=config["n_kpts"]).eval().to(device)
    elif extractor_type == 'sift':
        extractor = SIFT(max_num_keypoints=config["n_kpts"]).eval().to(device)
    else:
        raise ValueError("Unsupported feature extractor")

    return model, extractor

def loadModel(face_side):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  config = {
    'extractor': 'sift', # or 'superpoint',  # Or 'sift'
    'n_kpts': 1024,
    'n_layers': 9,
    'face_side': face_side,
    'metric': 'wassersteinH0',
    'no_match_thresh': 0.2
  }
  print("Loading LightGlue model...", face_side)
  model, extractor_obj = load_lightglue_model(config)
  print("LightGlue model loaded.")

  print("Creating inference engine...")
  # Initialize the inference class with the model and extractor
  inference_engine = LightGlueInference(model, extractor_obj, device)  # Updated to pass extractor_obj
  print("Inference engine created.")
  return inference_engine, config

inference_engineL, configL = loadModel('L')
inference_engineR, configR = loadModel('R')

def infer(name, image):
    # Extract face side from the image filename
    face_side = 'R' if ' R.' in name else 'L' if ' L.' in name else None
    if face_side is None:
        raise ValueError("Face side not found in the image filename")

    print("Infering...")
    inference_engine = inference_engineR
    config = configR
    
    if face_side == "L":
      inference_engine = inference_engineL
      config = configL

    top_n_matches = inference_engine.infer(image, config, topN=5)
    print("Infering done.")
    print("top_n_matches", top_n_matches)

    # Assuming top_n_matches returns (matched_image_names, confidences, ...)
    matched_image_info = list(zip(top_n_matches[0], top_n_matches[1]))

    print("matched_image_info", matched_image_info)

    return { 
       "prediction": matched_image_info
    }

    # Display the images
    # display_images(input_image_path, matched_image_info, base_image_path)


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/predict', methods=['POST'])
def predict():
    i = request.files['image'];
    image = Image.open(i.stream).convert('RGB')
    return jsonify(infer(i.filename, transforms.ToTensor()(image).unsqueeze_(0)))

@app.route('/')
def hello_world():
    return 'Hello, do you want to know where is your turtle?'

if __name__ == '__main__':
    app.run(debug=True)
