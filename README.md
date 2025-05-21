# Clothing Labeling: Manual and Algorithmic Labeling Toolkit

A lightweight Python project for experimenting with data labeling using clustering and classification algorithms such as **K-Means** and **K-Nearest Neighbors (KNN)**. Designed for educational purposes and quick prototyping on labeled datasets.

---

## Project Structure

```
├── KNN.py                 # K-Nearest Neighbors algorithm
├── Kmeans.py              # K-Means clustering algorithm
├── my_labeling.py         # Core logic for applying labels
├── TestCases_knn.py       # Unit tests for KNN
├── TestCases_kmeans.py    # Unit tests for K-Means
├── utils.py               # General utility functions
├── utils_data.py          # Data loading and preprocessing
├── gt.json                # Full ground truth annotations
├── gt_reduced.json        # Reduced/filtered ground truth (for testing)
├── train.zip              # Zipped training dataset (images)
├── test.zip               # Zipped test dataset (images)
├── README.md              # Project description and instructions
```

---

## Features

- Manual and automatic labeling strategies
- Unsupervised clustering (K-Means)
- Supervised classification (KNN)
- Test cases to validate correctness
- Dataset included as compressed `.zip` files

---

## Running the Project

To run the tests:

```bash
pytest TestCases_knn.py
pytest TestCases_kmeans.py
```

To run the scripts manually:

```bash
python Kmeans.py
python KNN.py
```

---

## Dataset Access

- Training and test images are included as `.zip` files: `train.zip` and `test.zip`
- To extract them:

```bash
unzip train.zip -d train/
unzip test.zip -d test/
```

Make sure the images are in the correct folders if used by any script.

---

## Requirements

- Python 3.6+
- NumPy
- pytest

Install with:

```bash
pip install numpy pytest
```

---

## Ground Truth

- `gt.json`: Full annotations
- `gt_reduced.json`: Lighter subset for development/testing

---

## ✍️ Author

This project was created as part of a data labeling and algorithm exploration exercise.

---

## License

MIT License (add if applicable)
