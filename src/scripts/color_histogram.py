import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def extract_features(image_path):
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            features = {
                'mean_red': np.mean(img_array[:, :, 0]),
                'mean_green': np.mean(img_array[:, :, 1]),
                'mean_blue': np.mean(img_array[:, :, 2]),
                'std_red': np.std(img_array[:, :, 0]),
                'std_green': np.std(img_array[:, :, 1]),
                'std_blue': np.std(img_array[:, :, 2]),
                'width': img.width,
                'height': img.height
            }
            return features
    except IOError:
        print(f"Error processing image {image_path}")
        return None


def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def create_dataframe(base_dir, categories):
    data = []
    for category in categories:
        category_path = os.path.join(base_dir, category)
        image_files = [f for f in os.listdir(category_path) if is_image_file(f)]
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            features = extract_features(image_path)
            if features:
                features['label'] = category.replace('_', ' ')  # Replace underscores with spaces
                data.append(features)
    return pd.DataFrame(data)

def plot_histogram(dataframe, feature, title, xlabel, bins=30):
    plt.figure(figsize=(10, 6))
    for label in dataframe['label'].unique():
        subset = dataframe[dataframe['label'] == label]
        plt.hist(subset[feature], bins=bins, alpha=0.5, label=f'{label} {feature}')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_multiple_histograms(dataframe, features):
    n_features = len(features)
    for i, feature_info in enumerate(features):
        plot_histogram(dataframe, **feature_info)