# Resolution-Agnostic Latent Distillation (RALD)
## A Theoretical Framework for Scale-Invariant OCR

### 1. The Core Problem: Representation Drift
Vision-Language Models (VLMs) like Qwen-VL operate on patches. When a model is fine-tuned on **word-crops**, the receptive fields of the Attention layers are saturated with high-frequency structural details (strokes, loops, terminals). 

When shifted to **page-level inference**, those same semantic characters occupy a fraction of the original token space. This causes **Representation Drift**: the latent vector representing a 'ব' at word-scale resides in a different manifold region than the vector for 'ব' at page-scale.

---

### 2. The Solution: Multi-Scale Consistency Loss (MSCL)

To bridge this gap, we propose a joint optimization objective that forces the model to learn a **Scale-Invariant Latent Manifold**. The objective is defined as:

$$ \mathcal{L}_{total} = \mathcal{L}_{ocr} + \lambda_{1}\mathcal{L}_{afa} + \lambda_{2}\mathcal{L}_{shl} $$

#### A. Anchored Feature Alignment (AFA)
Instead of treating word-crops and page-views as independent samples, we treat them as **Augmented Pairs** in the hidden space.

*   **Mechanism:** During training, we take a high-resolution word crop $x_{word}$ and its low-resolution "contextualized" version $x_{page}$ (where the word is embedded in a larger canvas).
*   **Objective:** We minimize the distance between the visual hidden states of the encoder.
*   **Formula:** 
    $$ \mathcal{L}_{afa} = \sum_{l \in Layers} \beta_l \cdot || z_{word}^{(l)} - \text{Proj}(z_{page}^{(l)}) ||_2^2 $$
    *   $z_{word}^{(l)}$: Hidden state from the word-crop.
    *   $z_{page}^{(l)}$: Hidden state from the corresponding region in the page-view.
    *   $\text{Proj}$: A learnable bottleneck that maps low-res features to the high-res manifold.

**Theory:** This forces the encoder to "hallucinate" high-frequency structural features even when the input pixels are blurry or small.

#### B. Semantic Hub Loss (SHL)
We use the model's own word-level mastery as a "Moving Teacher". 

*   **Mechanism:** Because the model already "knows" how to read word crops, we use the output probability distribution of the word-crop as the Ground Truth for the page-view.
*   **Objective:** Minimize the Kullback-Leibler (KL) Divergence between the two predictions.
*   **Formula:**
    $$ \mathcal{L}_{shl} = D_{KL} \left( P(y | x_{word}, \theta) \parallel P(y | x_{page}, \theta) \right) $$

**Theory:** This acts as a semantic anchor. It prevents **Catastrophic Forgetting** of word-level details while pulling the page-level reasoning into alignment with the word-level logic.

---

### 3. Theoretical Advantages

1.  **Scale Agnosticism:** The model no longer identifies a character by its absolute size in pixels, but by its relative position in the latent manifold.
2.  **Zero-Inference Overhead:** Unlike Tiling or Two-Stage pipelines, this framework modifies the **weights** during training. At inference time, you still only run a single forward pass on the page.
3.  **Data Efficiency:** It leverages your existing 100k word images by synthetically generating context, rather than requiring new, expensive manual page annotations.

### 4. Mathematical Goal
The ultimate goal of RALD is to minimize the **Mutual Information Gap** between different viewpoints of the same semantic content:

$$ I(Char; Image_{word}) \approx I(Char; Image_{page}) $$

By enforcing this identity in the loss function, the model becomes resident to layout shifts and resolution changes.
