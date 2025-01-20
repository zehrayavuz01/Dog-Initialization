# Optimizing CNNs with Difference of Gaussian Initialization

### [Project Report 🔗](#)

**Data Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

This project investigates the application of a biologically-inspired initialization method, Difference of Gaussian (DoG), in convolutional neural networks (CNNs). By exploring human retinal mechanisms, we aimed to improve performance through a novel kernel initialization approach. We tested this methodology using the CIFAR-10 dataset on a MobileNetV2 architecture.

---

## Objective

The study evaluates whether initializing CNN kernels with a DoG function—modeled after human retinal center-surround receptive fields—enhances performance across various hyperparameters.

---

## Methodology

1. **Initialization with DoG**:
   - Applied a Difference of Gaussian function to initialize convolutional kernels, mimicking the edge detection process in the human retina.
   - Balanced excitatory and inhibitory kernel initialization with a 50-50 distribution.

2. **Model Selection**:
   - Used the MobileNetV2 architecture for its computational efficiency.
   - Compared results to prior studies that used ConvNeXt Tiny for validation.

3. **Data Preprocessing**:
   - Dataset: CIFAR-10, comprising 60,000 RGB images (32x32 pixels).
   - Data split: 45,000 for training, 5,000 for validation, and 10,000 for testing.

4. **Hyperparameter Tuning**:
   - Investigated the influence of:
     - **γ (center-surround ratio)**: Values between 0.1 and 0.7.
     - **Share (proportion of DoG-initialized kernels)**: Values between 0.3 and 0.9.
   - Evaluated combinations of these hyperparameters using accuracy metrics.

5. **Training Process**:
   - Models trained using standard optimization techniques.
   - Compared DoG-initialized kernels to He Normal initialization as a baseline.

---

## Results

### Hyperparameter Grid Search
The heatmap below shows the performance (accuracy) across different combinations of the `gamma` and `share` parameters:

![Hyperparameter Grid Search](Dog-Initialization/heatmap.png)

### Accuracy Comparison
The table below highlights the accuracy achieved with He Normal Initialization compared to different configurations of DoG-initialized kernels:

![Accuracy Results](Dog-Initialization/result.png)

### Visualization of Kernels
The following visualization displays 50 randomly initialized kernels with the DoG initialization method (Share = 0.7, Gamma = 0.5):

![DoG Kernels](Dog-Initialization/share.png)

---

## Key Findings

- The DoG method did not significantly outperform traditional initialization methods on CIFAR-10.
- No consistent pattern was observed in performance changes across hyperparameter combinations.
- Results suggest the method may not generalize well across different datasets or simpler architectures.

---

## Future Work

Further studies are needed to refine this method for broader applications, focusing on:
- Kernel positional analysis for improved initialization.
- Larger and more complex datasets, such as ImageNet.
- Exploring other architectures to validate the generalizability of DoG initialization.

---

### Usage Instructions
- Clone this repository.
- Ensure you have Python 3.8+ and the necessary dependencies installed.
- Run the training script provided in the repository to experiment with the DoG initialization.
