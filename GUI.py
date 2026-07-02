import sys
import os
import random
import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QTextEdit, QLabel,
                               QProgressBar, QMessageBox, QFileDialog, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon

try:
    import pypdf
except ImportError:
    pypdf = None

#Константы
MIN_CHARS_WARNING = 100
MAX_FILE_CHARS = 50000
MAX_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 ГБ

#Стили(QSS)
LIGHT_THEME = """
QMainWindow, QWidget { background-color: #f4f5f7; color: #333333; }
#Sidebar { background-color: #ffffff; border-right: 1px solid #d5d5d5; }
#Sidebar QPushButton { text-align: left; padding: 12px 20px; font-size: 14px; font-weight: bold; border: none; background: transparent; border-radius: 8px; margin: 2px 10px; }
#Sidebar QPushButton:hover { background-color: #e0e0e0; }
#Sidebar QPushButton:checked { background-color: #2196F3; color: white; }
QTextEdit { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 8px; padding: 10px; font-size: 14px; color: #333333; }
QTextEdit:disabled { background-color: #f9f9f9; color: #757575; }
QPushButton.action-btn { background-color: #2196F3; color: white; font-weight: bold; border-radius: 6px; padding: 10px; }
QPushButton.action-btn:hover { background-color: #1976D2; }
QPushButton.secondary-btn { background-color: #e0e0e0; color: #333; font-weight: bold; border-radius: 6px; padding: 8px; }
QPushButton.secondary-btn:hover { background-color: #d5d5d5; }
#ResultBox { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 8px; }
QProgressBar { border: 1px solid #bbb; border-radius: 6px; text-align: center; font-weight: bold; color: black; background-color: #e0e0e0; height: 24px; }
QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }
QTableWidget { background-color: white; border: 1px solid #ccc; border-radius: 8px; gridline-color: #eee; }
QHeaderView::section { background-color: #f4f5f7; padding: 4px; border: 1px solid #ccc; font-weight: bold; }
"""

DARK_THEME = """
QMainWindow, QWidget { background-color: #121214; color: #e0e0e0; }
#Sidebar { background-color: #1e1e24; border-right: 1px solid #2a2a32; }
#Sidebar QPushButton { text-align: left; padding: 12px 20px; font-size: 14px; font-weight: bold; border: none; background: transparent; border-radius: 8px; margin: 2px 10px; color: #e0e0e0;}
#Sidebar QPushButton:hover { background-color: #2a2a32; }
#Sidebar QPushButton:checked { background-color: #2196F3; color: white; }
QTextEdit { background-color: #2a2a32; border: 1px solid #3a3a44; border-radius: 8px; padding: 10px; font-size: 14px; color: #ffffff; }
QTextEdit:disabled { background-color: #232329; color: #88888c; }
QPushButton.action-btn { background-color: #2196F3; color: white; font-weight: bold; border-radius: 6px; padding: 10px; }
QPushButton.action-btn:hover { background-color: #1976D2; }
QPushButton.secondary-btn { background-color: #32323d; color: white; font-weight: bold; border-radius: 6px; padding: 8px; }
QPushButton.secondary-btn:hover { background-color: #434352; }
#ResultBox { background-color: #1a1a1f; border: 1px solid #3a3a44; border-radius: 8px; }
QProgressBar { border: 1px solid #3a3a44; border-radius: 6px; text-align: center; font-weight: bold; color: white; background-color: #2a2a32; height: 24px; }
QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }
QTableWidget { background-color: #1e1e24; border: 1px solid #3a3a44; border-radius: 8px; gridline-color: #3a3a44; color: white; }
QHeaderView::section { background-color: #2a2a32; padding: 4px; border: 1px solid #3a3a44; font-weight: bold; color: white; }
QTableWidget QTableCornerButton::section { background-color: #2a2a32; }
"""

#Custom Поле(текст)
class PlainTextEdit(QTextEdit):
    def insertFromMimeData(self, source):
        if source.hasText():
            self.insertPlainText(source.text())

#Глав. окно
class AIDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Детектор текста")
        self.resize(1000, 700)
        self.is_dark_theme = True
        
        # Данные
        self.is_file_uploaded = False
        self.hidden_file_text = ""
        self.current_report_data = ""
        self.history_data = [] # Список для хранения истории проверок
        
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        #Боковая панель
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)

        # Профиль пользователя
        user_lbl = QLabel("👤 Пользователь")
        user_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        user_lbl.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(user_lbl)
        sidebar_layout.addSpacing(30)

        # Кнопки навигации
        self.btn_nav_check = QPushButton("📝 Проверка текста")
        self.btn_nav_check.setCheckable(True)
        self.btn_nav_check.setChecked(True)
        
        self.btn_nav_history = QPushButton("🗄️ История проверок")
        self.btn_nav_history.setCheckable(True)
        
        self.btn_nav_info = QPushButton("⚙️ Инструкция")
        self.btn_nav_info.setCheckable(True)

        self.nav_buttons = [self.btn_nav_check, self.btn_nav_history, self.btn_nav_info]
        
        for i, btn in enumerate(self.nav_buttons):
            sidebar_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, index=i: self.switch_page(index))

        sidebar_layout.addStretch()

        # Кнопка темы
        self.btn_theme = QPushButton("☀️ Светлая тема")
        self.btn_theme.clicked.connect(self.toggle_theme)
        sidebar_layout.addWidget(self.btn_theme)

        main_layout.addWidget(self.sidebar)

        #Основная область
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        #Создание страницы
        self.page_check = self.create_check_page()
        self.page_history = self.create_history_page()
        self.page_info = self.create_info_page()

        self.stacked_widget.addWidget(self.page_check)
        self.stacked_widget.addWidget(self.page_history)
        self.stacked_widget.addWidget(self.page_info)

    #Проверка текста
    def create_check_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Верхняя панель "загрузка файлов"
        top_layout = QHBoxLayout()
        btn_upload = QPushButton("📄 Загрузить файлы")
        btn_upload.setProperty("class", "secondary-btn")
        btn_upload.clicked.connect(self.upload_file)
        
        self.lbl_file_info = QLabel("Файл не загружен")
        self.lbl_file_info.setStyleSheet("color: #757575; font-style: italic;")
        
        btn_clear = QPushButton("🗑️ Очистить всё")
        btn_clear.setProperty("class", "secondary-btn")
        btn_clear.clicked.connect(self.clear_all)

        top_layout.addWidget(btn_upload)
        top_layout.addWidget(self.lbl_file_info)
        top_layout.addStretch()
        top_layout.addWidget(btn_clear)
        layout.addLayout(top_layout)

        # Текстовое поле
        self.text_area = PlainTextEdit()
        self.text_area.setPlaceholderText("Ввод текста...")
        layout.addWidget(self.text_area)

        # Кнопки действий
        action_layout = QHBoxLayout()
        btn_analyze = QPushButton("Проверить текст")
        btn_analyze.setProperty("class", "action-btn")
        btn_analyze.setMinimumHeight(45)
        btn_analyze.clicked.connect(self.analyze_text)
        
        btn_save = QPushButton("💾 Сохранить результат")
        btn_save.setProperty("class", "secondary-btn")
        btn_save.setMinimumHeight(45)
        btn_save.clicked.connect(self.save_result)
        
        action_layout.addWidget(btn_analyze, 2)
        action_layout.addWidget(btn_save, 1)
        layout.addLayout(action_layout)

        # Блок результатов
        self.result_box = QWidget()
        self.result_box.setObjectName("ResultBox")
        result_layout = QVBoxLayout(self.result_box)
        
        self.prob_label = QLabel("Ожидание текста...")
        self.prob_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.prob_label.setAlignment(Qt.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFormat("%p%") # Отображение процентов внутри
        
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.warning_label.setAlignment(Qt.AlignCenter)
        
        result_layout.addWidget(self.prob_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(self.warning_label)
        
        layout.addWidget(self.result_box)
        return page

    #История проверок
    def create_history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("🗄️ История проверок")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        # Фильтры
        filter_layout = QHBoxLayout()
        btn_f1 = QPushButton("За час")
        btn_f2 = QPushButton("Сегодня")
        btn_f3 = QPushButton("Неделя")
        btn_export = QPushButton("💾 Сохранить отчёт")
        
        for btn in [btn_f1, btn_f2, btn_f3, btn_export]:
            btn.setProperty("class", "secondary-btn")
            
        filter_layout.addWidget(btn_f1)
        filter_layout.addWidget(btn_f2)
        filter_layout.addWidget(btn_f3)
        filter_layout.addStretch()
        filter_layout.addWidget(btn_export)
        layout.addLayout(filter_layout)

        # Таблица
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Время", "Символов", "Вердикт", "ИИ (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True) # Включаем сортировку по клику на заголовок
        layout.addWidget(self.table)
        
        return page

    #Инструкция
    def create_info_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("⚙️ Инструкция")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        rules = (
            "AI Text Detector\n\n"
            "1. Чистая вставка: При копировании текста (Ctrl+V) программа автоматически\n"
            "   очищает его от лишних стилей, жирного шрифта, картинок и 'кривых' символов.\n\n"
            f"2. ⚠️ Лимиты: Не проверяйте слишком короткие тексты (менее {MIN_CHARS_WARNING} символов).\n"
            f"   Максимальный лимит для загрузки файлов: {MAX_FILE_CHARS} символов (или 1 ГБ веса).\n\n"
            "3. История: Все проверки за текущую сессию сохраняются во вкладке 'История'.\n"
            "   Вы можете отсортировать их, кликнув по заголовку столбца таблицы.\n\n"
            "4. Темы: Используйте кнопку в левом нижнем углу для переключения оформления.\n\n"

            "Разработчики:\n" 
            "1)Пойлов В.А.\n"
            "2)Мухатаев А.В.\n"
            "3)Черепов Д.А.\n"
            "4)Савинцев Д.А.\n"
            "5)Лушин С.Д.\n"
        )
        
        text_label = QLabel(rules)
        text_label.setFont(QFont("Segoe UI", 12))
        text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(text_label)
        layout.addStretch()
        
        return page

    #Логика интерфейса и тем
    def switch_page(self, index):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.stacked_widget.setCurrentIndex(index)

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME)
            self.btn_theme.setText("☀️ Светлая тема")
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.btn_theme.setText("🌙 Темная тема")

    #Логика работы детектора
    def clear_all(self):
        self.is_file_uploaded = False
        self.hidden_file_text = ""
        self.current_report_data = ""
        self.lbl_file_info.setText("Файл не загружен")
        self.text_area.setReadOnly(False)
        self.text_area.clear()
        self.prob_label.setText("Ожидание текста...")
        self.prob_label.setStyleSheet("")
        self.progress_bar.setValue(0)
        self.warning_label.setText("")

    def upload_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Текстовые и PDF файлы (*.txt *.pdf);;Текстовые файлы (*.txt);;PDF файлы (*.pdf)")
        if not filepath: return

        file_size_bytes = os.path.getsize(filepath)
        if file_size_bytes > MAX_SIZE_BYTES:
            QMessageBox.critical(self, "Ошибка", "Файл слишком большой. Максимальный размер — 1 ГБ.")
            return

        try:
            ext = os.path.splitext(filepath)[1].lower()
            filename = os.path.basename(filepath)
            extracted_text = ""

            if ext == '.txt':
                with open(filepath, 'r', encoding='utf-8') as file:
                    extracted_text = file.read(MAX_FILE_CHARS + 1)
            elif ext == '.pdf':
                if pypdf is None:
                    QMessageBox.critical(self, "Ошибка", "Нужна библиотека pypdf (pip install pypdf)")
                    return
                with open(filepath, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        if len(extracted_text) >= MAX_FILE_CHARS: break
                        text = page.extract_text()
                        if text: extracted_text += text + "\n"

            extracted_text = extracted_text[:MAX_FILE_CHARS].strip()
            if not extracted_text:
                QMessageBox.warning(self, "Внимание", "Не удалось извлечь текст.")
                return

            self.hidden_file_text = extracted_text
            self.is_file_uploaded = True
            self.lbl_file_info.setText(f"Загружен: {filename}")
            
            self.text_area.setReadOnly(False)
            self.text_area.clear()
            self.text_area.setText(f"[Текст файла скрыт для стабильности. Ввод заблокирован]")
            self.text_area.setReadOnly(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{str(e)}")

    def analyze_text(self):
        target_text = self.hidden_file_text if self.is_file_uploaded else self.text_area.toPlainText().strip()
        text_length = len(target_text)
        
        if text_length == 0:
            self.prob_label.setText("Пожалуйста, добавьте текст.")
            self.warning_label.setText("")
            return

        if text_length < MIN_CHARS_WARNING:
            self.warning_label.setText(f"⚠️ Текст слишком короткий ({text_length} симв.). Возможна ошибка.")
        else:
            self.warning_label.setText("")

        # Заглушка ML модели
        ai_probability = random.uniform(0, 100)
        self.progress_bar.setValue(int(ai_probability))
        
        if ai_probability >= 50.0:
            verdict = "ИИ"
            color = "#f44336" if self.is_dark_theme else "#d32f2f"
        else:
            verdict = "Человек"
            color = "#4CAF50" if self.is_dark_theme else "#388e3c"
            
        self.prob_label.setText(f"Похоже на: {verdict} ({ai_probability:.1f}%)")
        self.prob_label.setStyleSheet(f"color: {color};")
        
        # Меняем цвет полосы прогресса
        self.progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 5px; }}")
        
        # Данные для отчета
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.current_report_data = f"Вердикт: {verdict}\nВероятность ИИ: {ai_probability:.1f}%\nТекст:\n{target_text}"
        
        # Добавление в историю
        self.add_to_history(time_str, text_length, verdict, ai_probability)

    def add_to_history(self, time_str, length, verdict, prob):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        item_time = QTableWidgetItem(time_str)
        
        item_len = QTableWidgetItem()
        item_len.setData(Qt.EditRole, length)
        
        item_verd = QTableWidgetItem(verdict)
        
        item_prob = QTableWidgetItem()
        item_prob.setData(Qt.EditRole, round(prob, 1))

        self.table.setItem(row, 0, item_time)
        self.table.setItem(row, 1, item_len)
        self.table.setItem(row, 2, item_verd)
        self.table.setItem(row, 3, item_prob)

    def save_result(self):
        if not self.current_report_data:
            QMessageBox.information(self, "Нет данных", "Сначала проведите анализ текста.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить", "Report.txt", "Текстовые файлы (*.txt)")
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.current_report_data)
            QMessageBox.information(self, "Успех", "Сохранено!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AIDetectorApp()
    window.show()
    sys.exit(app.exec())
