# Clothing Labeling: Manual and Algorithmic Labeling Toolkit

A lightweight Python project for experimenting with data labeling using clustering and classification algorithms such as **K-Means** and **K-Nearest Neighbors (KNN)**. Designed for educational purposes and quick prototyping on labeled datasets.

---

## Project Structure

```
Etiquetador/
├── KNN.py                # Implementation of the K-Nearest Neighbors algorithm
├── Kmeans.py             # K-Means clustering for unsupervised labeling
├── my_labeling.py        # Core labeling logic
├── TestCases_knn.py      # Test cases for KNN
├── TestCases_kmeans.py   # Test cases for K-Means
├── utils.py              # General-purpose utilities
├── utils_data.py         # Data loading and preprocessing helpers
├── images/gt.json        # Ground truth annotations (JSON format)
```

---

## Features

- Manual and automatic labeling strategies
- Unsupervised clustering (KMeans)
- Supervised classification (KNN)
- Easy-to-run test cases
- Designed to be extendable with minimal dependencies

---

## Running the Project

You can run tests using:

```bash
pytest Etiquetador/TestCases_knn.py
pytest Etiquetador/TestCases_kmeans.py
```

Or run scripts individually:

```bash
python Etiquetador/Kmeans.py
python Etiquetador/KNN.py
```

---

## Requirements

- Python 3.6+
- NumPy
- scikit-learn (for real-world extension, if needed)
- pytest (for test cases)

To install dependencies:

```bash
pip install numpy pytest
```

---

## Ground Truth Format

The file `images/gt.json` contains ground truth labels in JSON format used for validating labeling performance.

---

## Author

This project was created as part of a data labeling and algorithm exploration exercise.

---

## 📜 License

MIT License (add if applicable)
