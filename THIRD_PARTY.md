# Third-party software and model sources

The image-preparation code loads upstream implementations and pretrained
checkpoints. Those checkouts and weights are obtained separately; their license
and attribution files remain with the upstream distributions. Exact configured
URLs, revisions where available, and generation options are in
`configs/paper/imagenet/models.json` and `configs/paper/imagenet/preparation.json`.

- StyleGAN-XL: <https://github.com/autonomousvision/stylegan_xl>
- EDM2: <https://github.com/NVlabs/edm2>
- VAR: <https://github.com/FoundationVision/VAR>
- DINOv2: <https://github.com/facebookresearch/dinov2>
- OpenAI CLIP: <https://github.com/openai/CLIP>

The preparation config also identifies the Inception detector and the specific
DDO/EDM2 checkpoints used in the study. Published FID values in the model table
are contextual values transcribed from the paper, not outputs of this code.

The numerical implementation uses NumPy and SciPy, optionally CuPy; reporting
uses pandas and Matplotlib. Refer to each dependency's distribution for its
license.
