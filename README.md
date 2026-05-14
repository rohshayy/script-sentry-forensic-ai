# Script-Sentry: Forensic Stylometry AI
### **Biometric Authentication via Deep Latent Representation and Multi-Anchor Enrollment**

## **Executive Summary**
Script-Sentry is a high-security Biometric Engine designed to distinguish between authentic human handwriting and high-quality forgeries. While traditional OCR (Optical Character Recognition) focuses on *character classification*, Script-Sentry analyzes **Stylometry**—the mathematical "DNA" of stroke patterns. By mapping handwriting into a deep latent space, the system can identify the biological "jitter" and unique stylistic signatures of an individual user.

---

## **Technical Architecture**

### **1. Bottleneck Feature Extraction**
The core engine is a Multi-Layer Perceptron (MLP) featuring a **Bottleneck Architecture**:
* **Input Layer (784 neurons):** Processes flattened 28x28 pixel intensity maps.
* **Latent Bottleneck (128 neurons):** Forces the model to compress the image into 128 essential stylistic values, effectively filtering out environmental noise and capturing the user's specific handwriting manifold.
* **Dual-Mode Inference:** * **Recognition Mode:** Identifies the character (Identity Match).
    * **Forensic Mode:** Extracts the 128-dimensional **Style Vector** for biometric comparison.

### **2. Forensic Manifold Learning**
The model was trained on the **EMNIST Balanced Dataset** (47 classes, ~112k samples). Through this training, the network learns a **Forensic Manifold**—a geometric space where authentic human handwriting styles cluster distinctly from machine-generated or "too perfect" strokes.

### **3. Triple-Anchor Verification Logic**
To account for natural human variance (since no one writes a letter exactly the same way twice), the system implements a **Multi-Anchor Enrollment** protocol:
* **Golden Standard Generation:** Users enroll with 3 authentic samples. The system calculates the **Mean Style Profile** of these vectors.
* **Cosine Similarity Metric:** Verification is determined by calculating the angular distance between a suspect sample and the Golden Standard. 
* **Dual-Gate Authentication:** A sample is only "Verified" if it matches the correct character identity **AND** meets a **92% Stylometric Similarity** threshold.

---

## **🚀 Execution Guide**

### **1. Environment Setup**
Install the required AI and imaging libraries:
```bash
pip install torch torchvision pillow matplotlib

```

### **2. Running Forgery Detection (Inference)**

The system compares a suspect image against a set of "Anchor" (Real) samples.

1. **Step A:** Place three authentic samples in the project folder as `real_1.png`, `real_2.png`, and `real_3.png`.
2. **Step B:** Place the suspect image you wish to test as `test_image.png`.
3. **Step C:** Execute the forensic engine:

```bash
python script_sentry_app_inference.py

```

* **Output:** The console generates a Forensic Report. If **Style Similarity < 92%**, a **FORGERY ALERT** is triggered even if the letter itself is correctly identified.

### **3. Re-Training the Model (Optional)**

To re-train the neural network on the EMNIST dataset from scratch:

```bash
python script_sentry_train.py

```

* **Expectation:** The script downloads the EMNIST dataset, executes 20 epochs, and generates a **Loss Convergence Graph**. Successful completion updates the weight file `scriptsentry_v1.pth`.

---

## **Technical Troubleshooting & Robustness**

**Input Normalization:** The `process_image` function handles input variance by converting all images to Grayscale ('L'), resizing to 28x28 pixels, and transforming them into Normalized Tensors. This ensures the 784-neuron input layer receives consistent data regardless of the original image's resolution.

## **Technical Stack**

* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** NumPy, Matplotlib, EMNIST, Pillow
* **Mathematical Core:** Cosine Similarity, Manifold Learning, Vector Space Embedding, SVD-based Feature Extraction

```

```
