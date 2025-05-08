## Computational Inscription Analysis Toolkit - Documentation Overview

**Last Updated:** May 8, 2025

**1. Project Goal**
This project aims to develop a comprehensive computational toolkit for the analysis of historical inscriptions. By leveraging computer vision and deep learning, it seeks to automate and enhance laborious manual tasks such as pen mark extraction, text detection, image restoration (inpainting), and character classification, ultimately aiding epigraphic research.

**2. Core Modules**

*   **Module 1: Pen Mark Extraction from Inscriptions**
    *   **Purpose:** To segment modern pen marks (annotations, tracings) from inscription images, either for separate analysis or to clean images for subsequent processing.
    *   **Method:** Utilizes OpenCV for image processing: grayscaling, global binary thresholding (inverted), and morphological opening to remove noise and isolate pen marks.
    *   **Key Parameters:** `threshold_value` (default 50), 3x3 kernel for opening.

*   **Module 2: MMOCR Custom Text Detection Pipeline**
    *   **Purpose:** To detect and localize text regions (bounding boxes/polygons) within stone inscription images.
    *   **Method:** Fine-tunes a DBNet++ model (specifically `dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py`) using the MMOCR toolbox. Employs a ResNet-50 backbone with FPN and o-CLIP pre-training influences for robust detection of arbitrarily shaped text.
    *   **Training:** 1000 epochs on a custom dataset of 50 inscription images (8 unique inscriptions).
    *   **Performance:** Achieved ~67% Recall and ~72% Precision (F1 ~69.4%).

*   **Module 3: DeepFill Image Inpainting Model**
    *   **Purpose:** To plausibly reconstruct missing or damaged regions within inscription images (e.g., cracks, lichen, faded characters).
    *   **Method:** Fine-tunes a pre-trained DeepFill v2 model (GAN-based, originally trained on CelebA-HQ) on the custom inscription dataset. Uses gated convolutions and contextual attention. Training involves artificial random rectangular masks (20-40% of image area).
    *   **Losses:** L1 reconstruction, WGAN-GP adversarial, and VGG-based perceptual loss.
    *   **Performance:** PSNR ~28.45 dB, SSIM ~0.912 on test set.

*   **Module 4: Classification using CRNN**
    *   **Purpose:** To classify segmented individual character images from inscriptions into one of 170 predefined classes.
    *   **Method:** Employs a Convolutional Recurrent Neural Network (CRNN). CNN blocks extract spatial features from 64x64 RGB character images, followed by Global Average Pooling. A Bidirectional LSTM processes these features, and a final Dense layer with softmax performs classification.
    *   **Training:** Trained for 50 epochs (early stopped around 26) with data augmentation (rotation, shift, zoom, etc.) and `ReduceLROnPlateau` callback.
    *   **Performance:** Achieved ~72.72% test accuracy.

**3. Overall Project Pipeline & Inter-module Synergy**
The modules are designed to function in a sequential pipeline:
1.  **Image Acquisition:** Obtain inscription image.
2.  **(Optional) Pen Mark Extraction (Module 1):** Isolate/remove modern annotations.
3.  **Text Detection (Module 2):** Identify text regions on the (cleaned) image.
4.  **(Conditional) Image Inpainting (Module 3):** If detected text regions are damaged, inpaint them to improve legibility/recognizability.
5.  **Character Segmentation (Implicit Step):** Segment detected text regions (original or inpainted) into individual characters (requires further development, not a standalone documented module here).
6.  **Character Classification (Module 4):** Classify each segmented character.
7.  **Post-processing & Interpretation:** Assemble classified characters into words/phrases for linguistic analysis.

**4. Key Technologies Used**
*   Python
*   OpenCV (for Module 1)
*   MMOCR (PyTorch-based, for Module 2 - DBNet++)
*   MMEditing (PyTorch-based, for Module 3 - DeepFill v2)
*   TensorFlow/Keras (for Module 4 - CRNN)
*   Deep Learning: CNNs, GANs, LSTMs

**5. Broader Future Directions**
*   **End-to-End System Development:** Integrate modules into a seamless pipeline with a user interface.
*   **Dataset Expansion & Curation:** Enlarge and diversify annotated inscription datasets.
*   **Uncertainty Quantification:** Provide confidence scores for predictions.
*   **Human-in-the-Loop Systems:** Allow expert correction to refine models (active learning).
*   **Multi-modal Analysis:** Incorporate 3D scan or RTI data.
*   **Deployment & Accessibility:** Make tools available to the epigraphic community (web platforms, plugins).