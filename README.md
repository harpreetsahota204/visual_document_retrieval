# Vision-Document Retrieval (VDR) Model for FiftyOne

This repository provides a FiftyOne integration for the Vision-Document Retrieval (VDR) model, enabling powerful text-to-image search capabilities within the FiftyOne ecosystem.

## Overview

The [Vision-Document Retrieval (VDR) model](https://huggingface.co/llamaindex/vdr-2b-v1) is a multimodal embedding model created by LlamaIndex and based on the Qwen2VL architecture. It transforms both images and text into a shared vector space, allowing for:

- Text-to-image search: Find images that match text descriptions
- Image-to-image similarity: Find visually similar images

**Specialized for Document Images**: This model excels at working with document images of all kinds, including:
- Scanned text documents
- Screenshots
- Charts and graphs
- Slides and presentations
- Forms and tables
- Technical diagrams
- Any images containing text

This implementation provides a simple way to use VDR within FiftyOne for semantic search and similarity-based exploration of your document image datasets.

## Features

- **Text-to-Image Similarity**: Search your images using natural language queries
- **Customizable Embeddings**: Adjust embedding dimension to balance accuracy and performance
- **Seamless FiftyOne Integration**: Works with FiftyOne's Brain tools for dataset exploration

## Installation

1. Register the model source repository with FiftyOne:

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/visual_document_retrieval", 
    overwrite=True
)
```

2. Download the model:

```python
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/visual_document_retrieval",
    model_name="llamaindex/vdr-2b-v1"
)
```

## Usage

### Loading the Model

```python
import fiftyone.zoo as foz

model = foz.load_zoo_model("llamaindex/vdr-2b-v1")
```

### Computing Embeddings and Building a Similarity Index

```python
import fiftyone.brain as fob

# Compute embeddings and build a similarity index
text_img_index = fob.compute_similarity(
    dataset,                        # Your FiftyOne dataset
    model="llamaindex/vdr-2b-v1",   # Model name, you can also use the multilingual model, vdr-2b-multi-v1
    brain_key="vdr_img",            # Key to store the results, can be whatever you want
)
```

### Finding Similar Images to a Text Query

```python
# Sort dataset by similarity to a text query
similar_samples = text_img_index.sort_by_similarity("your awesome text query!")

# Example document-specific queries:
# similar_samples = text_img_index.sort_by_similarity("invoices from 2023")
# similar_samples = text_img_index.sort_by_similarity("bar charts showing declining trends")
# similar_samples = text_img_index.sort_by_similarity("error messages containing API failures")

```

### Advanced Usage

#### Custom Embedding Dimension

```python
model = foz.load_zoo_model(
    "llamaindex/vdr-2b-v1", 
    embedding_dim=1024  # Reduce dimension for faster processing
)
```

#### Custom Prompts

```python
model = foz.load_zoo_model(
    "llamaindex/vdr-2b-v1",
    document_prompt="<|im_start|>system\nDescribe this image in detail.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown?<|im_end|>\n<|endoftext|>",
    query_prompt="<|im_start|>system\nFind images related to the query.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"
)
```

## Technical Details

- **Base Model**: Qwen2VL (LlamaIndex/VDR-2B) multimodal model
- **Embedding Dimension**: 2048 by default, can be reduced
- **Image Processing**: Automatically resizes images to match model requirements
- **Platform Requirements**: CUDA-capable GPU recommended for optimal performance
- **Memory Requirements**: Approximately 5GB GPU memory

# How It Works

The model computes embeddings by:

1. **For Images**:
   - Resizing the image to fit model requirements (multiples of 28px)
   - Passing the image through the vision encoder
   - Extracting and normalizing the embedding vector

2. **For Text**:
   - Formatting the text with a special prompt template
   - Using a dummy image (required by the model architecture)
   - Extracting and normalizing the embedding vector

3. **Similarity Calculation**:
   - Computing cosine similarity between normalized embeddings
   - Ranking results based on similarity scores

# Ideal Use Cases and Limitations

### Best For
- **Document Images**: Excels with any kind of document containing text
- **Screenshots**: Great for searching UI screenshots, web pages, or application interfaces
- **Charts and Diagrams**: Can understand and retrieve based on graphical data representations
- **Mixed Text/Visual Content**: Works well with slides, posters, or infographics

### Limitations
- While it can process natural photos, it's specialized for text-containing images
- Performance varies with more abstract or specialized concepts
- Processing large images or large batches requires significant GPU memory
- The dummy image requirement for text encoding is a limitation of the underlying model architecture

## License

This implementation is provided under the terms of the [base model's license (LlamaIndex/VDR-2B-V1), which is Apache 2.0](https://choosealicense.com/licenses/apache-2.0/).

## Acknowledgements

- This implementation builds on the Qwen2VL model architecture
- Integrated with FiftyOne for dataset exploration and management
- Thanks to LlamaIndex for the VDR-2B model release

# Citation

```bibtext
@misc{vdr-2b-v1,
  author = {LlamaIndex},
  title = {VDR-2B-v1: Vision-Document Retrieval Model},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/llamaindex/vdr-2b-v1}},
  note = {Accessed: 2025-04-15}
}
```