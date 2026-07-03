from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import fonts
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime

class Report:
    def __init__(self):
        # Элементы pdf файла
        self.elements = []
        # Данные для отчёта
        #self.data = []

        # ---------- РЕГИСТРАЦИЯ ШРИФТА (для русского текста) ----------
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

        styles = getSampleStyleSheet()

        # Основной стиль документа
        self.custom_style = ParagraphStyle(
            name='Custom',
            parent=styles['Normal'],
            fontName='DejaVu',
            fontSize=12,
        )

        # Стиль для заголовков документа
        self.title_style = ParagraphStyle(
            name='TitleCustom',
            parent=styles['Heading1'],
            fontName='DejaVu',
            fontSize=18,
            spaceAfter=20
        )

        # ---------- СОЗДАНИЕ ДОКУМЕНТА ----------
        self.doc = SimpleDocTemplate(
            "report.pdf",
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        pass
    # ---------- ЗАГОЛОВОК ----------
    def add_title(self, title : str):
        self.elements.append(Paragraph(title, self.title_style))
        self.elements.append(Paragraph(
            f"Дата создания: {datetime.now().strftime('%d.%m.%Y')}",
            self.custom_style
        ))
        self.elements.append(Spacer(1, 0.5 * inch))
        pass
    def create_table(self, table_data):
        table = Table(table_data, hAlign="LEFT")

        table.setStyle(TableStyle([
            # Цвет фона заголовка
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

            # Выравнивание
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

            # Шрифт
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),

            # Сетка
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        return table
    pass
    # ---------- Добавить одну запись отчёта ----------
    def add_record(self):

        pass
    # ---------- Сборка отчёта ----------
    def build(self):
        self.doc.build(self.elements)
        print("Отчёт создан\n")
        pass

    pass