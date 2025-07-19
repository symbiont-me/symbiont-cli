import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


class DocumentData:
    """Data structure to hold all information needed for document generation"""
    
    def __init__(self):
        self.query: str = ""
        self.collection_name: str = ""
        self.llm_response: Optional[str] = None
        self.search_results: List[Dict[str, Any]] = []
        self.config_info: Dict[str, Any] = {}
        self.processing_info: Dict[str, Any] = {}
        self.timestamp: datetime = datetime.now()


class DocumentGenerator:
    """Generates clean, professional documents from search results and LLM responses"""
    
    def __init__(self):
        self.data = DocumentData()
    
    def set_data(self, query: str, collection_name: str, search_results: List[Any], 
                 llm_response: Optional[str] = None, config_info: Optional[Dict] = None,
                 processing_info: Optional[Dict] = None):
        """Set the data for document generation"""
        self.data.query = query
        self.data.collection_name = collection_name
        self.data.llm_response = llm_response
        self.data.config_info = config_info or {}
        self.data.processing_info = processing_info or {}
        
        # Convert search results to standardized format
        self.data.search_results = []
        for result in search_results:
            result_data = {
                'content': getattr(result, 'page_content', str(result)),
                'metadata': getattr(result, 'metadata', {}),
                'relevance_score': getattr(result, 'metadata', {}).get('relevance_score', 'N/A')
            }
            self.data.search_results.append(result_data)
    
    def generate_document(self, output_path: str):
        """Generate document based on file extension"""
        file_path = Path(output_path)
        
        if file_path.suffix.lower() == '.docx':
            self._generate_word_document(output_path)
        elif file_path.suffix.lower() == '.pdf':
            self._generate_pdf_document(output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _generate_word_document(self, output_path: str):
        """Generate a Word document"""
        doc = Document()
        
        # Title Page
        title = doc.add_heading('Research Query Results', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph()
        
        # Query Information
        doc.add_heading('Query Information', level=1)
        
        info_table = doc.add_table(rows=4, cols=2)
        info_table.style = 'Table Grid'
        
        info_table.cell(0, 0).text = 'Query:'
        info_table.cell(0, 1).text = self.data.query
        
        info_table.cell(1, 0).text = 'Collection:'
        info_table.cell(1, 1).text = self.data.collection_name
        
        info_table.cell(2, 0).text = 'Generated:'
        info_table.cell(2, 1).text = self.data.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        info_table.cell(3, 0).text = 'Results Found:'
        info_table.cell(3, 1).text = str(len(self.data.search_results))
        
        doc.add_paragraph()
        
        # LLM Response Section
        if self.data.llm_response:
            doc.add_heading('AI Generated Summary', level=1)
            response_para = doc.add_paragraph(self.data.llm_response)
            response_para.style = 'Intense Quote'
            doc.add_paragraph()
        
        # Configuration Information
        if self.data.config_info:
            doc.add_heading('Configuration', level=1)
            for key, value in self.data.config_info.items():
                if isinstance(value, dict):
                    doc.add_paragraph(f"{key.title()}:", style='Heading 2')
                    for sub_key, sub_value in value.items():
                        doc.add_paragraph(f"  • {sub_key}: {sub_value}")
                else:
                    doc.add_paragraph(f"{key.title()}: {value}")
            doc.add_paragraph()
        
        # Search Results
        if self.data.search_results:
            doc.add_heading('Retrieved Documents', level=1)
            
            for i, result in enumerate(self.data.search_results, 1):
                doc.add_heading(f'Document {i}', level=2)
                
                # Metadata table
                if result['metadata']:
                    metadata_table = doc.add_table(rows=len(result['metadata']), cols=2)
                    metadata_table.style = 'Light Shading'
                    
                    for row, (key, value) in enumerate(result['metadata'].items()):
                        metadata_table.cell(row, 0).text = key.title()
                        metadata_table.cell(row, 1).text = str(value)
                
                # Content
                doc.add_paragraph('Content:', style='Heading 3')
                content_para = doc.add_paragraph(result['content'])
                content_para.style = 'Body Text'
                
                if i < len(self.data.search_results):
                    doc.add_page_break()
        
        # Save document
        doc.save(output_path)
    
    def _generate_pdf_document(self, output_path: str):
        """Generate a PDF document"""
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title
        story.append(Paragraph("Research Query Results", title_style))
        story.append(Spacer(1, 20))
        
        # Query Information
        story.append(Paragraph("Query Information", header_style))
        
        query_data = [
            ['Query:', self.data.query],
            ['Collection:', self.data.collection_name],
            ['Generated:', self.data.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Results Found:', str(len(self.data.search_results))]
        ]
        
        query_table = Table(query_data, colWidths=[1.5*inch, 4*inch])
        query_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(query_table)
        story.append(Spacer(1, 20))
        
        # LLM Response
        if self.data.llm_response:
            story.append(Paragraph("AI Generated Summary", header_style))
            story.append(Paragraph(self.data.llm_response, styles['BodyText']))
            story.append(Spacer(1, 20))
        
        # Configuration
        if self.data.config_info:
            story.append(Paragraph("Configuration", header_style))
            for key, value in self.data.config_info.items():
                if isinstance(value, dict):
                    story.append(Paragraph(f"<b>{key.title()}:</b>", styles['BodyText']))
                    for sub_key, sub_value in value.items():
                        story.append(Paragraph(f"  • {sub_key}: {sub_value}", styles['BodyText']))
                else:
                    story.append(Paragraph(f"<b>{key.title()}:</b> {value}", styles['BodyText']))
            story.append(Spacer(1, 20))
        
        # Search Results
        if self.data.search_results:
            story.append(Paragraph("Retrieved Documents", header_style))
            
            for i, result in enumerate(self.data.search_results, 1):
                # Document header
                doc_header = ParagraphStyle(
                    'DocumentHeader',
                    parent=styles['Heading3'],
                    fontSize=14,
                    spaceAfter=10,
                    textColor=colors.darkgreen
                )
                story.append(Paragraph(f"Document {i}", doc_header))
                
                # Metadata
                if result['metadata']:
                    metadata_rows = [[key.title(), str(value)] for key, value in result['metadata'].items()]
                    metadata_table = Table(metadata_rows, colWidths=[1.5*inch, 4*inch])
                    metadata_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(metadata_table)
                    story.append(Spacer(1, 10))
                
                # Content
                story.append(Paragraph("<b>Content:</b>", styles['BodyText']))
                # Split long content into smaller paragraphs
                content = result['content']
                if len(content) > 500:
                    # Split into chunks for better formatting
                    words = content.split()
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) > 500 and current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_length = len(word)
                        else:
                            current_chunk.append(word)
                            current_length += len(word) + 1
                    
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    for chunk in chunks:
                        story.append(Paragraph(chunk, styles['BodyText']))
                else:
                    story.append(Paragraph(content, styles['BodyText']))
                
                if i < len(self.data.search_results):
                    story.append(Spacer(1, 30))
        
        # Build PDF
        doc.build(story)