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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.model_path = self.parse_string(d, "model_path", default="llamaindex/vdr-2b-v1")
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        self.embedding_dim = self.parse_int(d, "embedding_dim", default=2048)
        self.document_prompt = self.parse_string(d, "document_prompt", default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>")
        self.query_prompt = self.parse_string(d, "query_prompt", default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>")


class VDRModel(fout.TorchImageModel, fom.PromptMixin):
    """
    Model for computing image-text similarity using Qwen2VL.

    This class provides functionality for:
    - Text-image similarity scoring
    - Text and image embedding generation

    Args:
        config: a :class:`VDRModelConfig`
    """

    def __init__(self, config):
        """Initialize the model.

        Args:
            config: a VDRModelConfig instance
        """
        super().__init__(config)

        self._text_features = None
        self.document_prompt = config.document_prompt
        self.query_prompt = config.query_prompt

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.

        This method returns ``False`` by default. Methods that can generate
        embeddings will override this via implementing the
        :class:`EmbeddingsMixin` interface.
        """
        return True

    def _load_model(self, config):
        """Load the model from disk.

        Args:
            config: VDRModelConfig instance

        Returns:
            loaded PyTorch model
        """
        # Define image resolution limits
        self.max_pixels = 768 * 28 * 28
        self.min_pixels = 1 * 28 * 28

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            use_fast=True,
            size={
                'shortest_edge': self.min_pixels,
                'longest_edge': self.max_pixels
                }
            )
        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device in ["cuda", "mps"] else None
        # Load model and processor
        logger.info(f"Loading model from {config.model_path}")

        if self.torch_dtype:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
            )

        # Set padding side
        self.model.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        
        # Store embedding dimension
        self.embedding_dim = config.embedding_dim

    # Helper functions for image resizing
    def _round_by_factor(self, number, factor=28):
        return round(number / factor) * factor

    def _ceil_by_factor(self, number, factor=28):
        return math.ceil(number / factor) * factor

    def _floor_by_factor(self, number, factor=28):
        return math.floor(number / factor) * factor

    def _smart_resize(self, height, width):
        h_bar = max(28, self._round_by_factor(height))
        w_bar = max(28, self._round_by_factor(width))
        
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self._floor_by_factor(height / beta)
            w_bar = self._floor_by_factor(width / beta)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta)
            w_bar = self._ceil_by_factor(width * beta)
            
        return w_bar, h_bar

    def _resize_image(self, image):
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL for resizing
            from torchvision.transforms.functional import to_pil_image
            pil_image = to_pil_image(image)
            new_size = self._smart_resize(pil_image.height, pil_image.width)
            return pil_image.resize(new_size)
        else:
            new_size = self._smart_resize(image.height, image.width)
            return image.resize(new_size)

    def _embed_prompts(self, prompts):
        """Internal method to embed text prompts.

        Args:
            prompts: list of text strings to embed

        Returns:
            torch.Tensor of text embeddings
        """
        # Create a dummy image for the text-only embedding
        dummy_image = Image.new('RGB', (56, 56))
        
        # Format prompts with the query template
        formatted_prompts = [self.query_prompt % p for p in prompts]
        
        # Process inputs
        inputs = self.processor(
            text=formatted_prompts,
            images=[dummy_image for _ in prompts],
            videos=None,
            padding='longest',
            return_tensors='pt'
        )
        
        if self._using_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Prepare inputs for generation
        cache_position = torch.arange(0, len(prompts), device=self.device if self._using_gpu else "cpu")
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )
        
        # Extract and normalize embeddings
        embeddings = output.hidden_states[-1][:, -1]
        return F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)

    def embed_prompts(self, prompts):
        """Embed multiple text prompts.

        Args:
            prompts: list of text strings to embed

        Returns:
            numpy array of embedded prompts
        """
        return self._embed_prompts(prompts).detach().cpu().numpy()

    def embed_prompt(self, prompt):
        """Embed a single text prompt.

        Args:
            prompt: text string to embed

        Returns:
            numpy array of embedded prompt
        """
        return self.embed_prompts([prompt])[0]

    def _get_text_features(self):
        """Get text features for the configured classes.

        Returns:
            torch.Tensor of text features
        """
        if self._text_features is None:
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            self._text_features = self._embed_prompts(prompts)

        return self._text_features

    def embed_images(self, imgs):
        """Embed multiple images.

        Args:
            imgs: list of images to embed

        Returns:
            numpy array of embedded images
        """
        # Resize images
        resized_images = [self._resize_image(img) for img in imgs]
        
        # Process inputs
        inputs = self.processor(
            text=[self.document_prompt] * len(imgs),
            images=resized_images,
            videos=None,
            padding='longest',
            return_tensors='pt'
        )
        
        if self._using_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Prepare inputs for generation
        cache_position = torch.arange(0, len(imgs), device=self.device if self._using_gpu else "cpu")
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, 
            cache_position=cache_position, 
            use_cache=False
            )
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True
            )
        
        # Extract and normalize embeddings
        embeddings = output.hidden_states[-1][:, -1]
        normalized = F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)
        
        return normalized.detach().cpu().numpy()

    def embed_image(self, img):
        """Embed a single image.

        Args:
            img: image to embed

        Returns:
            numpy array of embedded image
        """
        return self.embed_images([img])[0]

    def _get_class_logits(self, text_features, image_features):
        """Calculate similarity logits between text and image features.

        Args:
            text_features: torch.Tensor of text embeddings
            image_features: torch.Tensor of image embeddings

        Returns:
            tuple of (logits_per_image, logits_per_text) tensors
        """
        image_features = image_features / image_features.norm(
            dim=1, 
            keepdim=True
        )
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.

        Args:
            imgs: list of images to process

        Returns:
            model predictions processed by the output processor
        """
        # Get image dimensions for output processing
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].shape[-2:]
        else:
            width, height = imgs[0].size
        
        frame_size = (width, height)
        
        # Generate image embeddings
        image_embeddings = torch.tensor(self.embed_images(imgs), device=self.device if self._using_gpu else "cpu")
        
        # Get text features for classes
        text_features = self._get_text_features()
        
        # Calculate similarity scores
        output, _ = self._get_class_logits(text_features, image_embeddings)
        
        # Process output
        if hasattr(self, 'has_logits') and self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, frame_size, confidence_thresh=self.config.confidence_thresh
        )