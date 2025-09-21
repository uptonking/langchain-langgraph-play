from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/

# source = "https://arxiv.org/pdf/2501.17887"
path_base = "/Users/yaoo/Documents/01files/testpdf"
# source = path_base + "/pro-js-1page.pdf"
source = path_base + "/html5canvas1page-pic.pdf"

###### USING SIMPLE DEFAULT VALUES
# - GraniteDocling model
# - Using the transformers framework

# converter: DocumentConverter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(
#             pipeline_cls=VlmPipeline,
#         ),
#     }
# )

# doc = converter.convert(source=source).document

# print(doc.export_to_markdown())


###### USING MACOS MPS ACCELERATOR
# Demonstrates using MLX on macOS with MPS acceleration (macOS only).

pipeline_options = VlmPipelineOptions(vlm_options=vlm_model_specs.GRANITEDOCLING_MLX)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
