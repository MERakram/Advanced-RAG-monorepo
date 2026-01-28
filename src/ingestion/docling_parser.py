from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from pathlib import Path
from typing import List, Dict, Any

class DoclingParser:
    """
    Parses digital documents (PDF, DOCX, etc.) using Docling to preserve structure.
    """
    
    def __init__(self):
        # Use defaults which support PDF, DOCX, HTML, etc.
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a file and returns a structured dictionary.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"[Docling] Starting to parse: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Use convert_single for single file conversion (returns ConversionResult directly)
        result: ConversionResult = self.converter.convert_single(path)
        
        print(f"[Docling] Conversion complete, extracting content...")
        
        # New Docling API: use render_as_markdown() directly on result
        full_text = result.render_as_markdown()
        
        print(f"[Docling] Extracted {len(full_text)} characters of text")
        
        # Extract tables from assembled data if available
        tables = []
        if hasattr(result, 'assembled') and result.assembled:
            assembled = result.assembled
            if hasattr(assembled, 'tables') and assembled.tables:
                for table in assembled.tables:
                    if hasattr(table, 'export_to_dataframe'):
                        tables.append(table.export_to_dataframe().to_dict(orient="records"))
                print(f"[Docling] Extracted {len(tables)} tables")
        
        # Get page count from result if available
        page_count = len(result.pages) if hasattr(result, 'pages') and result.pages else 1

        return {
            "text": full_text,
            "tables": tables,
            "metadata": {
                "filename": path.name,
                "page_count": page_count,
                "origin": "docling"
            }
        }
