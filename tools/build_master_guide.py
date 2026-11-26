"""
Script Pembuat Dokumen Bedah Teknis Master:
1. TECHNICAL_DEEP_DIVE_MASTER_GUIDE.md (Markdown komprehensif)
2. TECHNICAL_DEEP_DIVE_MASTER_GUIDE.docx (Dokumen Word profesional berformat rapi)
"""

import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(WORKSPACE_DIR, "TECHNICAL_DEEP_DIVE_MASTER_GUIDE.md")
DOCX_PATH = os.path.join(WORKSPACE_DIR, "TECHNICAL_DEEP_DIVE_MASTER_GUIDE.docx")

print(f"Target Markdown: {MD_PATH}")
print(f"Target Word Doc: {DOCX_PATH}")
