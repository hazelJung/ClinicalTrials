from app.services.llm_client import get_llm_client
from app.services.prompts import get_system_prompt, get_user_prompt
from app.services.docx_template import generate_docx
import os
import uuid
import re


def clean_markdown_tables(text: str) -> str:
    """
    Post-process LLM output to fix malformed Markdown tables.
    Replaces excessively long dash sequences with proper short separators.
    """
    # Pattern to match table separator rows with excessive dashes
    # Matches rows like |:--------------------------------------|:------|
    def fix_separator_row(match):
        row = match.group(0)
        # Split by | and clean each cell
        cells = row.split('|')
        fixed_cells = []
        for cell in cells:
            cell = cell.strip()
            if not cell:
                fixed_cells.append('')
                continue
            # Check if it's a separator cell (contains only dashes and colons)
            if re.match(r'^:?-+:?$', cell):
                # Normalize to simple separator
                if cell.startswith(':') and cell.endswith(':'):
                    fixed_cells.append(':---:')
                elif cell.startswith(':'):
                    fixed_cells.append(':---')
                elif cell.endswith(':'):
                    fixed_cells.append('---:')
                else:
                    fixed_cells.append('---')
            else:
                fixed_cells.append(cell)
        return '|'.join(fixed_cells)
    
    # Find and fix table separator rows (rows with |---|---| pattern)
    # This regex matches table separator rows with potentially long dashes
    separator_pattern = r'\|[\s:]*-{3,}[\s:]*(?:\|[\s:]*-{3,}[\s:]*)+\|?'
    text = re.sub(separator_pattern, fix_separator_row, text)
    
    # Also fix any remaining long dash sequences (more than 10 dashes in a row)
    text = re.sub(r'-{10,}', '---', text)
    
    return text


class INDGeneratorService:
    def __init__(self):
        self.llm_client = get_llm_client()

    def generate_ind_draft(self, data: dict) -> dict:
        # 1. Prepare Prompts
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(data)

        # 2. Call LLM
        # Handle potential errors from LLM client
        try:
            generated_text = self.llm_client.generate(system_prompt, user_prompt)
        except Exception as e:
            # Fallback or re-raise
            # For now re-raise to be handled by API
            raise RuntimeError(f"LLM Generation failed: {str(e)}")

        # 3. Post-process to fix malformed markdown tables
        generated_text = clean_markdown_tables(generated_text)

        # 4. Generate DOCX
        # Ensure absolute path for file storage
        # Path logic assumes standard FastAPI structure: app/services/this_file.py
        # We want to save to PKSmart/data/generated_ind
        
        # Get directory of this file (services)
        services_dir = os.path.dirname(os.path.abspath(__file__))
        # Up to app
        app_dir = os.path.dirname(services_dir)
        # Up to PKSmart root
        root_dir = os.path.dirname(app_dir)
        
        output_dir = os.path.join(root_dir, "data", "generated_ind")
        os.makedirs(output_dir, exist_ok=True)

        # Sanitize filename
        drug_name = "".join(
            [
                c
                for c in data.get("drug_name", "Draft")
                if c.isalnum() or c in (" ", "_", "-")
            ]
        ).strip()
        filename = f"IND_Section8_{drug_name}_{uuid.uuid4().hex[:8]}.docx"
        file_path = os.path.join(output_dir, filename)

        generate_docx(generated_text, file_path)

        return {"text": generated_text, "file_path": file_path, "filename": filename}
