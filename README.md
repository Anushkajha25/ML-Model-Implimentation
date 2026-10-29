# ML Foundations

Notebooks from 2023–24, written while learning core machine-learning and deep-learning methods.
Each one covers a single model family on a public dataset. These are learning implementations,
not production code. For my main project, see
[gait-condition-classifier](https://github.com/Anushkajha25/gait-condition-classifier).

## Notebooks

| Notebook | Task | Dataset | Method | Result recorded in the notebook |
|---|---|---|---|---|
| [`hand_gesture_recognition`](hand_gesture_recognition.ipynb) | 10-class hand-gesture recognition | [LeapGestRecog](https://www.gti.ssr.upm.es/data/HandGesture_database) (GTI, UPM) | Small CNN in Keras on 50×50 grayscale images | 100% test accuracy\* |
| [`food101_transfer_learning`](food101_transfer_learning.ipynb) | Food image classification | [Food-101](https://huggingface.co/datasets/ethz/food101), 3- and 11-class subsets | Fine-tuned ResNet50 / InceptionV3 (Keras) | ~92% validation accuracy (3 and 11 classes) |
| [`svm_cats_vs_dogs`](svm_cats_vs_dogs.ipynb) | Cat vs. dog classification | Kaggle Dogs vs. Cats | Raw pixels → PCA → SVM | 43% accuracy (below chance; see notes) |
| [`house_price_linear_regression`](house_price_linear_regression.ipynb) | House-price regression | Housing CSVs for five Indian cities | Linear regression on area and bedrooms | R² = 0.08 |
| [`kmeans_customer_segmentation`](kmeans_customer_segmentation.ipynb) | Customer segmentation | Mall Customers (200 rows) | K-means with the elbow method, k = 5 | Clusters plotted (unsupervised) |

## Notes on the results

These are kept as recorded so the notebooks stay honest about what they show.

- \*The gesture data was split randomly by image, so frames from the same person and recording
  appear in both train and test. A 100% score mostly reflects that overlap. A subject-wise split
  would be the right test.
- The SVM uses flattened raw pixels, which carry little class information. Its accuracy is below
  the 50% chance level, so it is a baseline to beat, not a working classifier.
- The linear regression uses only two features, which explains the low R².

## Running

The notebooks were written for Google Colab and Kaggle. Datasets are not included; each notebook
expects its dataset at the path set in its first cells. Some of them use APIs that newer Keras
versions have removed, such as `keras.wrappers.scikit_learn`, so rerunning them unchanged needs an older
TensorFlow/Keras 2.x environment.
