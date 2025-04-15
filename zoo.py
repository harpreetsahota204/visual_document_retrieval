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

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VDRModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a Vision-Document Retrieval (VDR) Model.
    
    This config class extends TorchImageModelConfig to provide specific parameters
    needed for the Qwen2VL multimodal model used for text-image similarity search.
    
    Args:
        model_path (str): Path to the model's weights on disk or HuggingFace model ID.
            Defaults to "llamaindex/vdr-2b-v1".
        text_prompt (str): Optional baseline text prompt to use for classification.
            Defaults to "".
        embedding_dim (int): Dimension of the embeddings to use. Reducing this can
            save memory while maintaining most of the semantic information.
            Defaults to 2048.
        document_prompt (str): Template for processing images. Contains special tokens
            for the Qwen2 model format. Defaults to a standard prompt asking what's
            in the image.
        query_prompt (str): Template for processing text queries. Contains special tokens
            for the Qwen2 model format with a %s placeholder for the actual query.
            Defaults to a standard prompt format.
        query (str): Default query to use when none is provided.
            Defaults to "What is shown in this image?".
        max_pixels (int): Maximum number of pixels to process (limits memory usage).
            Defaults to 768 * 28 * 28.
        min_pixels (int): Minimum number of pixels to process (ensures adequate detail).
            Defaults to 1 * 28 * 28.
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        super().__init__(d)
        
        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="llamaindex/vdr-2b-v1")
        
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        
        # Dimension of embeddings to use (can be reduced to save memory)
        self.embedding_dim = self.parse_int(d, "embedding_dim", default=2048)
        
        # Template for image processing with special Qwen2 tokens
        self.document_prompt = self.parse_string(d, "document_prompt", default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>")
        
        # Template for text queries with special Qwen2 tokens
        self.query_prompt = self.parse_string(d, "query_prompt", default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>")
        
        # Default query when none is provided
        self.query = self.parse_string(d, "query", default="What is shown in this image?")
        
        # Maximum pixels to process (28 is patch size for the model)
        # 768 patches of 28x28 pixels is a reasonable upper limit
        self.max_pixels = 768 * 28 * 28
        
        # Minimum pixels to process (28 is patch size for the model)
        # At least 1 patch of 28x28 pixels is required
        self.min_pixels = 1 * 28 * 28


class VDRModel(fout.TorchImageModel, fom.PromptMixin):
    """Vision-Document Retrieval Model based on Qwen2VL.
    
    This model leverages a vision-language model (Qwen2VL) to create embeddings for
    both images and text in a shared vector space, enabling text-image similarity search.
    
    The model can:
    1. Embed images into a vector space
    2. Embed text queries into the same vector space
    3. Calculate similarity between images and text
    4. Support zero-shot classification by comparing image embeddings to class name embeddings
    
    It extends TorchImageModel for image processing capabilities and PromptMixin to
    enable text embedding capabilities.
    """
    def __init__(self, config):
        """Initialize the VDR model.
        
        Args:
            config: A VDRModelConfig instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Store config parameters as instance variables for easier access
        self._text_features = None  # Cached text features for classification
        self.embedding_dim = config.embedding_dim  # Dimension of embeddings to use
        self.document_prompt = config.document_prompt  # Template for image processing
        self.query_prompt = config.query_prompt  # Template for text queries
        self.max_pixels = config.max_pixels  # Maximum pixels to process
        self.min_pixels = config.min_pixels  # Minimum pixels to process
        
        # Storage for the last computed embeddings (needed for FiftyOne API)
        self._last_computed_embeddings = None

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.
        
        Returns:
            bool: Always True for this model as embedding generation is supported
        """
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts.
        
        Returns:
            bool: Always True for this model as text embedding is supported
        """
        return True

    def _load_model(self, config):
        """Load the model and processor from disk or HuggingFace.
        
        This method initializes both the processor (for tokenization and image
        preprocessing) and the model itself, configuring them for inference.

        Args:
            config: VDRModelConfig instance containing model parameters

        Returns:
            The loaded PyTorch model ready for inference
        """
        # Load processor first - handles tokenization and image preprocessing
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            use_fast=True,  # Use fast tokenizer implementation when available
            size={
                'shortest_edge': config.min_pixels,  # Minimum dimension
                'longest_edge': config.max_pixels    # Maximum dimension
                }
            ) 
        
        # Load the model itself
        logger.info(f"Loading model from {config.model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_path,
            trust_remote_code=True,  # Required for some models with custom code
            device_map=self.device,  # Map model to the appropriate device
        )
        
        # Configure padding for left-padding (important for Qwen2 models)
        # This ensures attention masks work correctly
        self.model.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        
        # Store embedding dimension for later use
        self.embedding_dim = config.embedding_dim
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        return self.model

    def round_by_factor(self, number: float, factor: int) -> int:
        """Round a number to the nearest multiple of a factor.
        
        Used for ensuring image dimensions are multiples of the patch size.
        
        Args:
            number: The number to round
            factor: The factor to round to (e.g., patch size)
            
        Returns:
            The rounded number as an integer
        """
        result = round(number / factor) * factor
        return result

    def ceil_by_factor(self, number: float, factor: int) -> int:
        """Round a number up to the nearest multiple of a factor.
        
        Used when we need to ensure the image is at least a certain size
        in multiples of the patch size.
        
        Args:
            number: The number to round up
            factor: The factor to round to (e.g., patch size)
            
        Returns:
            The ceiling-rounded number as an integer
        """
        result = math.ceil(number / factor) * factor
        return result

    def floor_by_factor(self, number: float, factor: int) -> int:
        """Round a number down to the nearest multiple of a factor.
        
        Used when we need to ensure the image doesn't exceed a certain size
        in multiples of the patch size.
        
        Args:
            number: The number to round down
            factor: The factor to round to (e.g., patch size)
            
        Returns:
            The floor-rounded number as an integer
        """
        result = math.floor(number / factor) * factor
        return result

    def smart_resize(self, height: int, width: int) -> tuple[int, int]:
        """Resize image dimensions to be compatible with the model's requirements.
        
        This method ensures:
        1. Dimensions are multiples of the patch size (28)
        2. Total pixels are within min/max bounds
        3. Aspect ratio is preserved as much as possible
        
        Args:
            height: Original image height
            width: Original image width
            
        Returns:
            A tuple of (width, height) dimensions to resize to
        """
        # Round dimensions to multiples of patch size (28)
        h_bar = max(28, self.round_by_factor(height, 28))
        w_bar = max(28, self.round_by_factor(width, 28))

        # If image is too large, scale it down while preserving aspect ratio
        if h_bar * w_bar > self.max_pixels:
            # Calculate scaling factor to get total pixels under max_pixels
            beta = math.sqrt((height * width) / self.max_pixels)
            # Apply scaling and ensure dimensions are multiples of patch size
            h_bar = self.floor_by_factor(height / beta, 28)
            w_bar = self.floor_by_factor(width / beta, 28)
        # If image is too small, scale it up while preserving aspect ratio
        elif h_bar * w_bar < self.min_pixels:
            # Calculate scaling factor to get total pixels above min_pixels
            beta = math.sqrt(self.min_pixels / (height * width))
            # Apply scaling and ensure dimensions are multiples of patch size
            h_bar = self.ceil_by_factor(height * beta, 28)
            w_bar = self.ceil_by_factor(width * beta, 28)
            
        # Return width, height (note order is reversed from input)
        return w_bar, h_bar

    def _resize_image(self, image):
        """Resize an image to be compatible with the model's requirements.
        
        Handles both PIL images and PyTorch tensors, converting as needed.
        
        Args:
            image: A PIL image or PyTorch tensor to resize
            
        Returns:
            A resized PIL image
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL for resizing
            from torchvision.transforms.functional import to_pil_image
            pil_image = to_pil_image(image)
            # Calculate new dimensions with smart_resize
            new_size = self.smart_resize(pil_image.height, pil_image.width)
            return pil_image.resize(new_size)
        else:
            # Directly resize PIL image
            new_size = self.smart_resize(image.height, image.width)
            return image.resize(new_size)

    def _get_text_features(self):
        """Get or compute text features for the model's classification.
        
        This method caches the result for efficiency in repeated calls.
        
        Returns:
            numpy array: Text features as a numpy array for classification
        """
        # Check if text features are already computed and cached
        if self._text_features is None:
            # Use the query_prompt template and inject the text_prompt into it
            prompt = self.query_prompt % self.config.text_prompt
            # Compute and cache the text features
            self._text_features = self._embed_prompts([prompt])
        
        # Return the cached features
        return self._text_features
    
    def _embed_prompts(self, prompts):
        """Embed text prompts for similarity search.
        
        Follows the approach used in the native implementation of the model,
        using a dummy image as required by the multimodal architecture.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Embeddings for the prompts as numpy arrays
        """
        # Create a small dummy image required by the multimodal model architecture
        # This is needed even for text-only queries due to the model's design
        dummy_image = Image.new('RGB', (56, 56))
        
        # Format prompts with the query template
        formatted_prompts = [self.query_prompt % p for p in prompts]
        
        # Process inputs with both text and dummy images
        # The model expects both modalities even if we're only using text
        inputs = self.processor(
            text=formatted_prompts,
            images=[dummy_image for _ in prompts],
            videos=None,  # Explicitly set videos to None
            padding='longest',  # Use longest sequence for padding
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)  # Move to the appropriate device
        
        # Prepare inputs for generation as required by the model
        cache_position = torch.arange(0, len(prompts), device=self.device)
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, cache_position=cache_position, use_cache=False)
        
        # Run model inference with no gradient computation needed
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True  # Request hidden states for embeddings
            )
        
        # Extract embeddings from the last hidden state's final token
        # This captures the contextual representation of the entire input
        embeddings = output.hidden_states[-1][:, -1]
        
        # Normalize embeddings to unit length for cosine similarity computation
        normalized = F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)
        
        # Return as CPU numpy array for FiftyOne compatibility
        return normalized.detach().cpu().numpy()

    def embed_prompt(self, prompt):
        """Embed a single text prompt.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: Embedding for the prompt
        """
        # Format prompt with the template (done inside _embed_prompts)
        # Embed the single prompt by calling _embed_prompts with a list
        embeddings = self._embed_prompts([prompt])
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_prompts(self, prompts):
        """Embed multiple text prompts.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Embeddings for the prompts
        """
        # Directly call _embed_prompts which handles batch processing
        return self._embed_prompts(prompts)

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)
            
        Returns:
            numpy array: Embeddings for the images
        """
        # Process the batch of images with the document prompt
        # Each image uses the same document prompt template
        inputs = self.processor(
            text=[self.config.document_prompt] * len(imgs),
            images=[self._resize_image(img) for img in imgs],  # Resize images to model requirements
            videos=None,  # Explicitly set videos to None
            padding='longest',  # Use longest sequence for padding
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)  # Move to the appropriate device

        # Prepare inputs for generation as required by the model
        cache_position = torch.arange(0, len(imgs), device=self.device)
        inputs = self.model.prepare_inputs_for_generation(
            **inputs, 
            cache_position=cache_position, 
            use_cache=False)
        
        # Run model inference with no gradient computation needed
        with torch.no_grad():
            output = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True  # Request hidden states for embeddings
            )

        # Extract embeddings from the last hidden state's final token
        embeddings = output.hidden_states[-1][:, -1]

        # Normalize embeddings to unit length for cosine similarity computation
        normalized = F.normalize(embeddings[:, :self.embedding_dim], p=2, dim=-1)
        
        # Store the embeddings for later use by get_embeddings()
        self._last_computed_embeddings = normalized
    
        # Return as CPU numpy array for FiftyOne compatibility
        return normalized.detach().cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Implementation of TorchEmbeddingsMixin.embed() method.
        
        Args:
            img: PIL image or PyTorch tensor to embed
            
        Returns:
            numpy array: Embedding for the image
        """
        # Convert single image to a list for batch processing
        if isinstance(img, torch.Tensor):
            imgs = [img]
        else:
            imgs = [img]
        
        # Embed the single image using the batch method
        embeddings = self.embed_images(imgs)
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Implementation of TorchEmbeddingsMixin.embed_all() method.
        
        Args:
            imgs: List of images to embed (PIL images or PyTorch tensors)
            
        Returns:
            numpy array: Embeddings for the images
        """
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.
        
        Returns:
            numpy array: The last computed embeddings
            
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
            
        # Return the stored embeddings as a CPU numpy array
        return self._last_computed_embeddings.detach().cpu().numpy()

    def _get_class_logits(self, text_features, image_features):
        """Calculate similarity scores between text and image features.
        
        For normalized vectors, the dot product equals cosine similarity,
        which measures how similar the content of images and text are.
        
        Args:
            text_features: Text embeddings (normalized)
            image_features: Image embeddings (normalized)
            
        Returns:
            tuple: (logits_per_image, logits_per_text) similarity matrices
        """
        # Use torch.no_grad() to prevent gradient computation
        with torch.no_grad():
            # Compute dot product similarity between normalized vectors
            # This is equivalent to cosine similarity
            logits_per_image = torch.mm(image_features, text_features.T)
            
            # Transpose the similarity matrix to get text-to-image similarities
            logits_per_text = logits_per_image.T
            
            # Return both similarity matrices
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.
        
        Used for zero-shot classification by comparing image embeddings
        to text embeddings of class names.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            numpy array: Probability distribution over classes
        """
        # Get image embeddings
        image_embeddings = self.embed_images(imgs)
        
        # Get text embeddings for classes
        text_features = self._get_text_features()
        
        # Calculate similarity between images and text
        logits_per_image, logits_per_text = self._get_class_logits(text_features, image_embeddings)
        
        # Convert similarities to probabilities using softmax
        probs = F.softmax(logits_per_text, dim=-1)
        
        # Return as CPU numpy array
        return probs.cpu().numpy()