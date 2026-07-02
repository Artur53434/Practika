import sys
import os
import random
import datetime
import webbrowser
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QTextEdit, QLabel,
                               QProgressBar, QMessageBox, QFileDialog, QDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

try:
    import pypdf
except ImportError:
    pypdf = None

#Константы лимитов
MIN_CHARS_WARNING = 100
MAX_FILE_CHARS = 50000
MAX_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 ГБ

#Горячие клавиши и функции выделения
def select_all_text(event=None):
    """Выделение всего текста"""
    text_area.tag_add(tk.SEL, "1.0", tk.END)
    text_area.mark_set(tk.INSERT, "1.0")
    text_area.see(tk.INSERT)
    return 'break'

class InstructionWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Инструкция по использованию")
        self.setFixedSize(500, 500)
        
        layout = QVBoxLayout()
        
        title = QLabel("📖 Правила пользования")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        rules = (
            "AI Text Detector:\n\n"
            f"1. ⚠️ ВАЖНО О ЛИМИТАХ: Не проверяйте слишком короткие тексты\n"
            f"   (менее {MIN_CHARS_WARNING} символов). Результат может быть неточным.\n"
            f"   Максимальный лимит для загрузки: {MAX_FILE_CHARS} символов.\n\n"
            "2. Ввод текста: Вставьте текст вручную или загрузите файл.\n"
            "   При загрузке файла поле текста блокируется, а сам\n"
            "   текст скрывается для ускорения работы интерфейса.\n\n"
            "3. Сохранение: После проверки вы можете сохранить\n"
            "   результат и отчет в виде .txt файла на ваш компьютер.\n\n"

            "Разработчики:\n" 
            "1)Пойлов В.А.\n"
            "2)Мухатаев А.В.\n"
            "3)Черепов Д.А.\n"
            "4)Савинцев Д.А.\n"
            "5)Лушин С.Д.\n"
        )
        
        text_label = QLabel(rules)
        text_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(text_label)
        
        btn_ok = QPushButton("Понятно")
        btn_ok.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 4px;")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)


class AIDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Детектор сгенерированного текста")
        self.resize(750, 700)
        
        # Состояния
        self.is_file_uploaded = False
        self.hidden_file_text = ""
        self.current_report_data = ""
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        #Верхняя панель инструментов
        top_layout = QHBoxLayout()
        
        self.btn_paste = QPushButton("📋 Вставить")
        self.btn_paste.clicked.connect(self.paste_clipboard)
        
        self.btn_clear = QPushButton("🗑️ Очистить")
        self.btn_clear.clicked.connect(self.clear_all)
        
        self.btn_upload = QPushButton("📄 Загрузить Файл")
        self.btn_upload.clicked.connect(self.upload_file)
        
        self.btn_info = QPushButton("❓ Инструкция")
        self.btn_info.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_info.clicked.connect(self.show_instructions)
        
        top_layout.addWidget(self.btn_paste)
        top_layout.addWidget(self.btn_clear)
        top_layout.addWidget(self.btn_upload)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_info)
        
        main_layout.addLayout(top_layout)

        #Текстовое поле
        self.text_area = QTextEdit()
        self.text_area.setFont(QFont("Segoe UI", 11))
        self.text_area.setPlaceholderText("Вставьте текст сюда или загрузите файл...")
        main_layout.addWidget(self.text_area)

        #Кнопки действий
        action_layout = QHBoxLayout()
        
        self.btn_analyze = QPushButton("Проверить текст")
        self.btn_analyze.setStyleSheet("background-color: #2196F3; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.btn_analyze.clicked.connect(self.analyze_text)
        
        self.btn_save = QPushButton("💾 Сохранить результат")
        self.btn_save.setStyleSheet("background-color: #607D8B; color: white; font-size: 12px; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_result)
        
        action_layout.addStretch()
        action_layout.addWidget(self.btn_analyze)
        action_layout.addWidget(self.btn_save)
        action_layout.addStretch()
        
        main_layout.addLayout(action_layout)

        #Блок результатов
        result_widget = QWidget()
        result_widget.setStyleSheet("background-color: white; border: 1px solid #e0e0e0; border-radius: 8px;")
        result_layout = QVBoxLayout(result_widget)
        
        self.prob_label = QLabel("Вероятность генерации ИИ: 0%")
        self.prob_label.setFont(QFont("Segoe UI", 11))
        self.prob_label.setAlignment(Qt.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; border-radius: 5px; height: 22px; font-weight: bold; color: black; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }
        """)
        
        self.result_label = QLabel("Ожидание текста...")
        self.result_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.result_label.setStyleSheet("color: #757575;")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        self.warning_label = QLabel("")
        self.warning_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.warning_label.setStyleSheet("color: #FF9800;")
        self.warning_label.setAlignment(Qt.AlignCenter)
        
        result_layout.addWidget(self.prob_label)
        result_layout.addWidget(self.progress_bar)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.warning_label)
        
        main_layout.addWidget(result_widget)

        #Ссылка на сайт Серёги и его подвала
        self.link_btn = QPushButton("🌐 Перейти на сайт с приложением")
        self.link_btn.setCursor(Qt.PointingHandCursor)
        self.link_btn.setStyleSheet("color: #1976D2; text-decoration: underline; border: none; background: transparent;")
        self.link_btn.clicked.connect(lambda: webbrowser.open_new("https://artur53434.github.io/Practika/"))
        main_layout.addWidget(self.link_btn, alignment=Qt.AlignCenter)

    #Функции
    def paste_clipboard(self):
        if not self.text_area.isReadOnly():
            self.text_area.paste()

    def clear_all(self):
        self.is_file_uploaded = False
        self.hidden_file_text = ""
        self.current_report_data = ""
        
        self.text_area.setReadOnly(False)
        self.text_area.clear()
        
        self.result_label.setText("Ожидание текста...")
        self.result_label.setStyleSheet("color: #757575;")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #bbb; border-radius: 5px; height: 22px; font-weight: bold; color: black; } QProgressBar::chunk { background-color: #4CAF50; border-radius: 5px; }")
        self.prob_label.setText("Вероятность генерации ИИ: 0%")
        self.warning_label.setText("")

    def show_instructions(self):
        dlg = InstructionWindow(self)
        dlg.exec()

    def upload_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "Текстовые и PDF файлы (*.txt *.pdf);;Текстовые файлы (*.txt);;PDF файлы (*.pdf)"
        )
        
        if not filepath:
            return

        file_size_bytes = os.path.getsize(filepath)
        if file_size_bytes > MAX_SIZE_BYTES:
            QMessageBox.critical(self, "Ошибка", "Файл слишком большой. Максимальный размер - 1 ГБ.")
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
                    QMessageBox.critical(self, "Ошибка", "Для чтения PDF нужна библиотека pypdf.\nВ терминале: pip install pypdf")
                    return
                with open(filepath, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        if len(extracted_text) >= MAX_FILE_CHARS: break
                        text = page.extract_text()
                        if text: extracted_text += text + "\n"

            extracted_text = extracted_text[:MAX_FILE_CHARS].strip()

            if not extracted_text:
                QMessageBox.warning(self, "Внимание", "Не удалось извлечь текст или файл пуст.")
                return

            self.hidden_file_text = extracted_text
            self.is_file_uploaded = True
            
            self.text_area.setReadOnly(False)
            self.text_area.clear()
            self.text_area.setText(f"📄 Файл успешно загружен: {filename}\n({file_size_bytes // 1024} КБ)\n\n[Текст скрыт для стабильной работы интерфейса. Ввод заблокирован]")
            self.text_area.setReadOnly(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл:\n{str(e)}")

    def analyze_text(self):
        target_text = self.hidden_file_text if self.is_file_uploaded else self.text_area.toPlainText().strip()
        text_length = len(target_text)
        
        if text_length == 0:
            self.result_label.setText("Пожалуйста, добавьте текст.")
            self.result_label.setStyleSheet("color: #d32f2f;")
            self.warning_label.setText("")
            return
            
        if text_length > MAX_FILE_CHARS:
            target_text = target_text[:MAX_FILE_CHARS]
            text_length = len(target_text)

        if text_length < MIN_CHARS_WARNING:
            self.warning_label.setText(f"⚠️Текст слишком короткий ({text_length} симв.). Высокая вероятность ошибки!")
        else:
            self.warning_label.setText("")

        #Заглушка ML модели
        ai_probability = random.uniform(0, 100)
        self.progress_bar.setValue(int(ai_probability))
        self.prob_label.setText(f"Вероятность генерации ИИ: {ai_probability:.1f}%")
        
        if ai_probability >= 50.0:
            verdict = "ИИ"
            color = "#d32f2f"
            self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #bbb; border-radius: 5px; height: 22px; font-weight: bold; color: black; } QProgressBar::chunk { background-color: #d32f2f; border-radius: 5px; }")
        else:
            verdict = "человека"
            color = "#388e3c"
            self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #bbb; border-radius: 5px; height: 22px; font-weight: bold; color: black; } QProgressBar::chunk { background-color: #388e3c; border-radius: 5px; }")
            
        self.result_label.setText(f"Похоже на: {verdict} ({ai_probability:.1f}%)")
        self.result_label.setStyleSheet(f"color: {color};")
        
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_report_data = (
            f"Отчёт анализа текста\n"
            f"Дата: {date_str}\n"
            f"Объем текста: {text_length} символов\n\n"
            f"Результат:\n"
            f"Вердикт: Похоже на {verdict}\n"
            f"Вероятность генерации ИИ: {ai_probability:.1f}%\n\n"
            f"Исследуемый текст\n{target_text}\n"
        )

    def save_result(self):
        if not self.current_report_data:
            QMessageBox.information(self, "Нет данных", "Сначала проведите анализ текста.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "AI_Analysis_Resultt.txt", "Текстовые файлы (*.txt)")
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.current_report_data)
                QMessageBox.information(self, "Отлично", "Результат успешно сохранен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AIDetectorApp()
    window.show()
    sys.exit(app.exec())
root.mainloop()
