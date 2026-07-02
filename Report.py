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


# ---------- РЕГИСТРАЦИЯ ШРИФТА (для русского текста) ----------
pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

styles = getSampleStyleSheet()

custom_style = ParagraphStyle(
    name='Custom',
    parent=styles['Normal'],
    fontName='DejaVu',
    fontSize=12,
)

title_style = ParagraphStyle(
    name='TitleCustom',
    parent=styles['Heading1'],
    fontName='DejaVu',
    fontSize=18,
    spaceAfter=20
)


# ---------- СОЗДАНИЕ ДОКУМЕНТА ----------
doc = SimpleDocTemplate(
    "report.pdf",
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

elements = []

# ---------- ЗАГОЛОВОК ----------
elements.append(Paragraph("Отчёт по продажам", title_style))
elements.append(Paragraph(
    f"Дата создания: {datetime.now().strftime('%d.%m.%Y')}",
    custom_style
))
elements.append(Spacer(1, 0.5 * inch))


# ---------- ДАННЫЕ ----------
data = [
    ["№", "Товар", "Количество", "Цена", "Сумма"],
    [1, "Ноутбук", 2, 50000, 100000],
    [2, "Мышь", 5, 1500, 7500],
    [3, "Клавиатура", 3, 3000, 9000],
]

# Подсчёт итогов
total_sum = sum(row[4] for row in data[1:])
data.append(["", "", "", "ИТОГО:", total_sum])


# ---------- СОЗДАНИЕ ТАБЛИЦЫ ----------
table = Table(data, hAlign="LEFT")

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

    # Подсветка итоговой строки
    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
    ('SPAN', (0, -1), (2, -1)),  # объединяем ячейки
]))

elements.append(table)


# ---------- СБОРКА PDF ----------
doc.build(elements)

print("PDF-отчёт создан: report.pdf")