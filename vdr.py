import logging
import os
from packaging.version import Version
import warnings
import math

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VDRModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`VDRModel`.

    Args:
        model_path: the path to the model's weights on disk
        text_prompt: the text prompt to use, e.g., ``"A photo of"``
        classes: the list of classes to use for zero-shot prediction
        embedding_dim: the dimension of the embeddings to use
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: a dictionary containing the configuration parameters
        """
        super().__init__(d)

        self.text_prompt = self.parse_string(d, "text_prompt", default="A photo of")
        self.embedding_dim = self.parse_int(d, "embedding_dim", default=2048)


class VDRModel(fout.TorchImageModel, fom.PromptMixin):
    def __init__(self, config):
        super().__init__(config)
        self._text_features = None
        self.document_prompt = config.document_prompt
        self.query_prompt = config.query_prompt
        # Define image resolution limits
        self.max_pixels = 768 * 28 * 28
        self.min_pixels = 1 * 28 * 28

    def _embed_prompts(self, prompts):
        """Internal method to embed text prompts following native implementation.

        Args:
            prompts: list of text strings to embed

        Returns:
            torch.Tensor of text embeddings
        """
        print(f"Embedding {len(prompts)} prompts...")
        
        # Create a dummy image for the text-only embedding
        dummy_image = Image.new('RGB', (56, 56))
        
        inputs = self.processor(
            text=[self.query_prompt % x for x in prompts],
            images=[dummy_image for _ in prompts],
            videos=None,
            padding='longest',
            return_tensors='pt'
        )
        
        if self._using_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        cache_position = torch.arange(0, len(prompts), device=self.device if self._using_gpu else "cpu")
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )
        
        embeddings = output.hidden_states[-1][:, -1]
        return F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)

    def embed_images(self, imgs):
        """Embed multiple images following native implementation.

        Args:
            imgs: list of images to embed

        Returns:
            numpy array of embedded images
        """
        print(f"Embedding {len(imgs)} images...")
        
        # Resize images
        resized_images = [self._resize_image(img) for img in imgs]
        print("Images resized")
        
        inputs = self.processor(
            text=[self.document_prompt] * len(imgs),
            images=resized_images,
            videos=None,
            padding='longest',
            return_tensors='pt'
        )
        
        if self._using_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        cache_position = torch.arange(0, len(imgs), device=self.device if self._using_gpu else "cpu")
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)
        
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )
        
        embeddings = output.hidden_states[-1][:, -1]
        normalized = F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)
        
        print("Image embedding complete")
        return normalized.detach().cpu().numpy()

    def _get_text_features(self):
        """Get text features using document prompt.

        Returns:
            torch.Tensor of text features
        """
        if self._text_features is None:
            # Use the document prompt as the text feature
            self._text_features = self._embed_prompts([self.config.text_prompt])

        return self._text_features

    def _predict_all(self, imgs):
        """Run prediction on a batch of images."""
        print(f"\n=== _predict_all ===")
        print(f"Number of images: {len(imgs)}")
        
        # Get image dimensions for output processing
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].shape[-2:]
        else:
            width, height = imgs[0].size
        
        frame_size = (width, height)
        print(f"Frame size: {frame_size}")
        
        # Generate image embeddings
        print("Generating image embeddings...")
        image_embeddings = torch.tensor(self.embed_images(imgs), device=self.device if self._using_gpu else "cpu")
        print(f"Image embeddings shape: {image_embeddings.shape}")
        
        # Get text features
        print("Getting text features...")
        text_features = self._get_text_features()
        print(f"Text features shape: {text_features.shape}")
        
        # Calculate similarity scores
        output, _ = self._get_class_logits(text_features, image_embeddings)
        print(f"Output logits shape: {output.shape}")
        
        return self._output_processor(
            output, frame_size, confidence_thresh=self.config.confidence_thresh
        )