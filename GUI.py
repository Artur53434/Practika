import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import webbrowser
import os
import datetime

try:
    import pypdf
except ImportError:
    pypdf = None

#Константы лимитов
MIN_CHARS_WARNING = 100     # Порог для min текста
MAX_FILE_CHARS = 50000      # max кол-во символов для анализа

#Глобальные переменные состояния
inst_win = None
is_file_uploaded = False
hidden_file_text = ""
current_report_data = ""    # Хранит текст отчета для сохранения

#Горяие клавиши и функции выделения
def select_all_text(event=None):
    """Выделение всего текста"""
    text_area.tag_add(tk.SEL, "1.0", tk.END)
    text_area.mark_set(tk.INSERT, "1.0")
    text_area.see(tk.INSERT)
    return 'break'

def paste_clipboard(event=None):
    """Вставка текста (заблокирована, если загружен файл)"""
    if text_area.cget("state") == tk.DISABLED:
        return 'break'
        
    try:
        if text_area.tag_ranges(tk.SEL):
            text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
        clipboard_text = root.clipboard_get()
        text_area.insert(tk.INSERT, clipboard_text)
    except tk.TclError:
        messagebox.showwarning("Внимание", "Буфер обмена пуст или содержит не текст.")
    return 'break'

def clear_all():
    """Полная очистка поля, сброс состояний файлов и результатов"""
    global is_file_uploaded, hidden_file_text, current_report_data
    
    is_file_uploaded = False
    hidden_file_text = ""
    current_report_data = ""
    
    # Разблокируем поле и очищаем
    text_area.config(state=tk.NORMAL)
    text_area.delete("1.0", tk.END)
    
    # Сбрасываем UI
    result_label.config(text="Ожидание текста...", fg="#757575")
    prob_var.set(0)
    prob_label.config(text="Вероятность генерации ИИ: 0%")
    model_attribution_label.config(text="")
    warning_label.config(text="")

#Функции загрузки
def upload_file():
    global is_file_uploaded, hidden_file_text
    
    filepath = filedialog.askopenfilename(
        title="Выберите файл (.txt или .pdf)",
        filetypes=[("Текстовые и PDF файлы", "*.txt *.pdf"), ("Текстовые файлы", "*.txt"), ("PDF файлы", "*.pdf")]
    )
    
    if not filepath:
        return

    #Проверка на 1Гб
    file_size_bytes = os.path.getsize(filepath)
    MAX_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 ГБ
    
    if file_size_bytes > MAX_SIZE_BYTES:
        messagebox.showerror("Ошибка", "Файл слишком большой. Максимальный размер - 1 ГБ.")
        return

    try:
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        extracted_text = ""
        #логика чтения файла
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as file:
                extracted_text = file.read(MAX_FILE_CHARS + 1) # Читаем только до лимита символов
                
        elif ext == '.pdf':
            if pypdf is None:
                messagebox.showerror("Ошибка", "Для чтения PDF нужна библиотека pypdf.")
                return
            with open(filepath, 'rb') as file:
                reader = pypdf.PdfReader(file)
                # Читаем страницы, пока не наберем лимит символов
                for page in reader.pages:
                    if len(extracted_text) >= MAX_FILE_CHARS: break
                    text = page.extract_text()
                    if text: extracted_text += text
        
        # Обрезаем, если вдруг прочитали лишнее
        extracted_text = extracted_text[:MAX_FILE_CHARS].strip()

        if not extracted_text:
            messagebox.showwarning("Внимание", "Не удалось извлечь текст или файл пуст.")
            return

        # Сохраняем в скрытую переменную и блокируем поле
        hidden_file_text = extracted_text
        is_file_uploaded = True
        
        text_area.config(state=tk.NORMAL)
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, f"📄 Файл: {filename}\n({file_size_bytes // 1024 // 1024} МБ)\n\n[Текст скрыт для более стабильной работы]")
        text_area.config(state=tk.DISABLED)
            
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

#ИНСТРУКЦИЯ
def show_instructions():
    global inst_win
    if inst_win is not None and inst_win.winfo_exists():
        inst_win.lift()
        inst_win.focus_force()
        return
        
    inst_win = tk.Toplevel(root)
    inst_win.title("Инструкция по использованию")
    inst_win.geometry("620x550")
    inst_win.configure(bg="white")
    inst_win.resizable(False, False)
    
    rules = (
        "AI Text Detector:\n\n"
        f"1. Важно про лимиты(!) Не проверяйте слишком\n"
        f"   короткие тексты (менее {MIN_CHARS_WARNING} символов). Модели не хватает\n"
        f"   контекста, и результат может быть крайне неточным.\n"
        f"   Максимальный лимит для загрузки: {MAX_FILE_CHARS} символов.\n\n"
        "2. Ввод текста: Вставьте текст вручную или загрузите файл.\n"
        "   При загрузке файла поле текста блокируется, а сам\n"
        "   текст скрывается для ускорения работы интерфейса.\n\n"
        "3. Атрибуция: Поставьте галочку, чтобы узнать вероятную\n"
        "   модель ИИ (GPT-4, Claude и т.д.).\n\n"
        "4. Сохранение: После проверки вы можете сохранить\n"
        "   результат и отчет в виде .txt файла на ваш компьютер.\n\n"

        "Разработчики:\n" 
        "1)Пойлов В.А.\n"
        "2)Мухатаев А.В.\n"
        "3)Черепов Д.А.\n"
        "4)Савинцев Д.А.\n"
        "5)Лушин С.Д.\n"
    )
    
    tk.Label(inst_win, text="📖 Правила пользования", font=("Segoe UI", 12, "bold"), bg="white").pack(pady=(15, 5))
    tk.Label(inst_win, text=rules, font=("Segoe UI", 10), bg="white", justify=tk.LEFT).pack(padx=20, pady=5)
    tk.Button(inst_win, text="Понятно", command=inst_win.destroy, bg="#2196F3", fg="white", relief=tk.FLAT, padx=20).pack(pady=10)

#Анализ текста
def analyze_text():
    global current_report_data
    
    # Берем текст либо из скрытой переменной файла, либо из поля ввода
    if is_file_uploaded:
        target_text = hidden_file_text
    else:
        target_text = text_area.get("1.0", tk.END).strip()
        
    text_length = len(target_text)
    
    if text_length == 0:
        result_label.config(text="Пожалуйста, добавьте текст.", fg="#d32f2f")
        warning_label.config(text="")
        return
        
    # Дополнительная защита лимитов при ручном вводе
    if text_length > MAX_FILE_CHARS:
        target_text = target_text[:MAX_FILE_CHARS]
        text_length = len(target_text)

    # Проверка на слишком короткий текст
    if text_length < MIN_CHARS_WARNING:
        warning_label.config(text=f"⚠️ Текст слишком короткий ({text_length} симв.). Высокая вероятность ошибки.", fg="#FF9800")
    else:
        warning_label.config(text="")

    #ЗАГЛУШКА ML МОДЕЛИ
    ai_probability = random.uniform(0, 100)
    prob_var.set(ai_probability)
    prob_label.config(text=f"Вероятность генерации ИИ: {ai_probability:.1f}%")
    
    model_attribution_label.config(text="")
    likely_model = "Не определено"
    
    if ai_probability >= 50.0:
        verdict = "ИИ"
        color = "#d32f2f"
        
        if check_model_var.get():
            models = ["ChatGPT (GPT-4)", "Claude 3", "YandexGPT", "GigaChat", "Llama 3"]
            likely_model = random.choice(models)
            model_attribution_label.config(text=f"Вероятная модель: {likely_model}", fg="#1976D2")
    else:
        verdict = "человека"
        color = "#388e3c"
        
    result_label.config(text=f"Похоже на: {verdict}", fg=color)
    
    # Формируем данные для сохранения отчета
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_report_data = (
        f"Отчёт Анализа Текста\n"
        f"Дата: {date_str}\n"
        f"Объем текста: {text_length} символов\n\n"
        f"Результат:\n"
        f"Вердикт: Похоже на {verdict}\n"
        f"Вероятность генерации ИИ: {ai_probability:.1f}%\n"
    )
    if check_model_var.get() and verdict == "ИИ":
        current_report_data += f"Вероятная модель БЯМ: {likely_model}\n"
        
    current_report_data += f"\nИсследуемый текст\n{target_text}\n"

#Сохранение результата
def save_result():
    if not current_report_data:
        messagebox.showinfo("Нет данных", "Сначала проведите анализ текста, чтобы сохранить результат.")
        return
        
    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Текстовый файл", "*.txt")],
        title="Сохранить результат проверки",
        initialfile="AI_Analysis_Result.txt"
    )
    
    if filepath:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(current_report_data)
            messagebox.showinfo("Успех", "Результат успешно сохранен!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

def open_website(event):
    webbrowser.open_new("https://tvoi-sait-ai.github.io/")

#Оформление и запуск
root = tk.Tk()
root.title("Детектор сгенерированного текста")
root.geometry("700x720")
root.configure(bg="#f4f5f7")

style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", thickness=20, background="#4CAF50", troughcolor="#e0e0e0")

#Верхняя панель инструментов
top_frame = tk.Frame(root, bg="#f4f5f7")
top_frame.pack(pady=(15, 5), fill=tk.X, padx=30)

#Работа с буфером и полем
tools_frame = tk.Frame(top_frame, bg="#f4f5f7")
tools_frame.pack(side=tk.LEFT)
ttk.Button(tools_frame, text="📋 Вставить", command=paste_clipboard).pack(side=tk.LEFT, padx=(0, 5))
ttk.Button(tools_frame, text="🗑️ Очистить", command=clear_all).pack(side=tk.LEFT, padx=(0, 15))

#Файлы и Инструкция
ttk.Button(top_frame, text="📄 Загрузить Файл", command=upload_file).pack(side=tk.LEFT)
tk.Button(top_frame, text="❓ Инструкция", command=show_instructions, bg="#FF9800", fg="white", relief=tk.FLAT).pack(side=tk.RIGHT)

#Текстовое поле
text_frame = tk.Frame(root, bg="#f4f5f7")
text_frame.pack(padx=30, pady=5, fill=tk.BOTH)

text_area = tk.Text(text_frame, height=10, width=70, font=("Segoe UI", 11), relief=tk.FLAT)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Привязка горячих клавиш
text_area.bind("<Control-a>", select_all_text)
text_area.bind("<Control-A>", select_all_text)
text_area.bind("<Control-f>", select_all_text)
text_area.bind("<Control-F>", select_all_text)
text_area.bind("<Control-v>", paste_clipboard)
text_area.bind("<Control-V>", paste_clipboard)
text_area.bind("<Control-m>", paste_clipboard)
text_area.bind("<Control-M>", paste_clipboard)
text_area.bind("<<Paste>>", paste_clipboard)

scrollbar = ttk.Scrollbar(text_frame, command=text_area.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scrollbar.set)

#галочка атрибуции модели
check_model_var = tk.BooleanVar()
check_model_btn = tk.Checkbutton(root, text="Определить вероятную модель ИИ", 
                                 variable=check_model_var, bg="#f4f5f7", font=("Segoe UI", 10), activebackground="#f4f5f7")
check_model_btn.pack(pady=5)

#Кнопки управления 
action_frame = tk.Frame(root, bg="#f4f5f7")
action_frame.pack(pady=5)

tk.Button(action_frame, text="Проверить текст", font=("Segoe UI", 12, "bold"), 
          bg="#2196F3", fg="white", relief=tk.FLAT, padx=20, pady=8, cursor="hand2", command=analyze_text).pack(side=tk.LEFT, padx=10)

tk.Button(action_frame, text="💾 Сохранить результат", font=("Segoe UI", 10, "bold"), 
          bg="#607D8B", fg="white", relief=tk.FLAT, padx=15, pady=6, cursor="hand2", command=save_result).pack(side=tk.LEFT, padx=10)

#Результаты + предупреждения
result_frame = tk.Frame(root, bg="white", highlightbackground="#e0e0e0", highlightthickness=1)
result_frame.pack(pady=10, fill=tk.X, padx=30)

prob_var = tk.DoubleVar()
prob_label = tk.Label(result_frame, text="Вероятность генерации ИИ: 0%", font=("Segoe UI", 11), bg="white")
prob_label.pack(pady=(10, 5))

ttk.Progressbar(result_frame, variable=prob_var, maximum=100, length=500, style="TProgressbar").pack(pady=5)

result_label = tk.Label(result_frame, text="Ожидание текста...", font=("Segoe UI", 16, "bold"), bg="white", fg="#757575")
result_label.pack(pady=(5, 0))

model_attribution_label = tk.Label(result_frame, text="", font=("Segoe UI", 11, "bold"), bg="white")
model_attribution_label.pack(pady=(0, 5))

warning_label = tk.Label(result_frame, text="", font=("Segoe UI", 10, "bold"), bg="white", fg="#FF9800")
warning_label.pack(pady=(0, 10))

#Ссылка на сайт СЕРЁГИННОГО ПОДВАЛА
link_label = tk.Label(root, text="🌐 Перейти на сайт с приложением", font=("Segoe UI", 10, "underline"), bg="#f4f5f7", fg="#1976D2", cursor="hand2")
link_label.pack(side=tk.BOTTOM, pady=15)
link_label.bind("<Button-1>", open_website)

root.mainloop()
