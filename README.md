# MTC-AIC3-BCI-Competition
## Model Architecture

Our core model builds on the proven EEGNet V4 backbone, then enhances it with two lightweight Transformer heads to further boost temporal feature learning.

### Base: EEGNet V4  
EEGNet V4 is specifically designed for efficient, accurate EEG signal decoding. Key components include:  
- **Squeeze-and-Excitation (SE) blocks**  
  Dynamically recalibrate channel-wise feature responses, enabling the network to focus on the most informative EEG channels.  
- **Depthwise Convolution**  
  Learns spatial filters across channels independently, capturing electrode‑to‑electrode patterns with minimal parameters.  
- **Separable Convolution**  
  Decomposes standard convolutions into a depthwise spatial step followed by a pointwise channel‑mixing step, for both efficiency and expressivity.  
- **Attention Layer**  
  Allows the network to reweight temporal or spatial features, further improving its ability to pick out salient EEG dynamics.  

This combination ensures a compact yet powerful feature extractor, tailored for brain–computer‑interface tasks.

### Extension: Dual Transformer Heads  
To capture longer-range temporal dependencies beyond the receptive field of standard convolutions, we append two small Transformer encoders after the EEGNet V4 backbone:  
1. **Transformer Head 1**  
   - Embeds the convolutional feature maps into a sequence of tokens  
   - Applies multi‑head self‑attention to model medium‑range temporal structure  
2. **Transformer Head 2**  
   - Takes the output of Head 1, re-embeds it, and applies a second self‑attention block  
   - Further refines long‑term dependencies before the final classification layer  

Both heads use lightweight architectures (few layers, small hidden dimensions) to keep overall model complexity manageable.

### How It Works  
1. **Input**: Raw EEG epochs (channels × time samples)  
2. **Feature Extraction**:  
   - **SE block** squeezes global channel statistics and excites important channels  
   - **Depthwise & separable convolutions** extract spatial–temporal features  
   - **Attention** layer reweights salient features  
3. **Sequence Formation**: Convolutional outputs are reshaped into a token sequence  
4. **Temporal Modeling**: Two cascaded Transformer encoders capture medium‑ and long‑range dependencies  
5. **Classification**: A final fully connected layer maps to class probabilities  

This hybrid CNN–Transformer design leverages the strengths of both local feature extraction and global temporal modeling, yielding state‑of‑the‑art performance on motor imagery and SSVEP tasks.  
