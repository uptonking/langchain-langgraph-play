from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# https://docling-project.github.io/docling/examples/minimal/

path_base = "/Users/yaoo/Documents/01files/testpdf"

# source = "https://arxiv.org/pdf/2408.09869"
# source = "/Users/yaoo/Documents/html5canvas1page.pdf"
# source = path_base + "/JavaScript高级程序设计-1page.pdf"
source = path_base + "/pro-js-1page.pdf"

artifacts_path = "/Users/yaoo/.cache/docling/models"

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
# doc_converter = DocumentConverter()
doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)


doc = doc_converter.convert(source).document

print(doc.export_to_markdown())
