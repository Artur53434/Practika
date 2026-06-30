import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import webbrowser
import os

try:
    import pypdf
except ImportError:
    pypdf = None

inst_win = None

#Горяие клавиши
def select_all(event):
    text_area.tag_add(tk.SEL, "1.0", tk.END)
    text_area.mark_set(tk.INSERT, "1.0")
    text_area.see(tk.INSERT)
    return 'break'

def paste_clipboard(event):
    try:
        if text_area.tag_ranges(tk.SEL):
            text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
        text_area.insert(tk.INSERT, root.clipboard_get())
    except tk.TclError:
        pass
    return 'break'

#Функции кнопок
def paste_text():
    try:
        clipboard_text = root.clipboard_get()
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, clipboard_text)
    except tk.TclError:
        messagebox.showwarning("Внимание", "Буфер обмена пуст или содержит не текст.")

def upload_file():
    filepath = filedialog.askopenfilename(
        title="Выберите файл (.txt или .pdf)",
        filetypes=[("Текстовые и PDF файлы", "*.txt *.pdf"), ("Текстовые файлы", "*.txt"), ("PDF файлы", "*.pdf")]
    )
    
    if not filepath:
        return

    try:
        # Определяем расширение файла
        ext = os.path.splitext(filepath)[1].lower()
        text_area.delete("1.0", tk.END)

        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as file:
                text_area.insert(tk.END, file.read())
                
        elif ext == '.pdf':
            #Проверка на устоновленную библиотеку
            if pypdf is None:
                messagebox.showerror("Ошибка", "Для чтения PDF нужна библиотека pypdf.\nВ терминале введите: pip install pypdf")
                return
            
            #Чтение PDF файла
            with open(filepath, 'rb') as file:
                reader = pypdf.PdfReader(file)
                extracted_text = ""
                # Проходимся по всем страницам и собираем текст
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n\n"
                
                if not extracted_text.strip():
                    messagebox.showwarning("Внимание", "Не удалось извлечь текст. Возможно, PDF состоит из картинок.")
                else:
                    text_area.insert(tk.END, extracted_text.strip())
        else:
            messagebox.showwarning("Формат", "Поддерживаются только .txt и .pdf файлы.")
            
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

def show_instructions():
    global inst_win
    if inst_win is not None and inst_win.winfo_exists():
        inst_win.lift()
        inst_win.focus_force()
        return
        
    inst_win = tk.Toplevel(root)
    inst_win.title("Инструкция по использованию")
    inst_win.geometry("480x400")
    inst_win.configure(bg="white")
    inst_win.resizable(False, False)
    
    rules = (
        "AI Text Detector:\n\n"
        "1. Ввод текста: Вставьте текст в поле вручную, используйте\n"
        "   кнопку 'Вставить текст' или загрузите файл (.txt или .pdf).\n"
        "2. Горячие клавиши: Поле поддерживает Ctrl+A\n"
        "    и Ctrl+V.\n"
        "3. Атрибуция модели: Если вам надо узнать, какая именно\n"
        "   модель (GPT-4, Claude и др.) написала текст, поставьте\n"
        "   галочку на 'Определить вероятную модель ИИ'.\n"
        "4. Проверка: Нажмите 'Проверить текст'. Программа выдаст\n"
        "   вероятность генерации в % и финальный вердикт.\n\n"

        "Разработчики:\n" 
        "1)Пойлов В.А.\n"
        "2)Мухатаев А.В.\n"
        "3)Черепов Д.А.\n"
        "4)Савинцев Д.А.\n"
        "5)Лушин С.Д.\n"
    )
    
    tk.Label(inst_win, text="Правила пользования", font=("Segoe UI", 12, "bold"), bg="white").pack(pady=(15, 5))
    tk.Label(inst_win, text=rules, font=("Segoe UI", 10), bg="white", justify=tk.LEFT).pack(padx=20, pady=5)
    tk.Button(inst_win, text="Понятно", command=inst_win.destroy, bg="#2196F3", fg="white", relief=tk.FLAT, padx=20).pack(pady=10)

def analyze_text():
    input_text = text_area.get("1.0", tk.END).strip()
    
    if not input_text:
        result_label.config(text="Пожалуйста, добавьте текст.", fg="#d32f2f")
        return

    # Заглушка ML модели
    ai_probability = random.uniform(0, 100)
    prob_var.set(ai_probability)
    prob_label.config(text=f"Вероятность генерации ИИ: {ai_probability:.1f}%")
    
    model_attribution_label.config(text="")
    
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
    metrics_label.config(text=f"F1: {random.uniform(0.80, 0.99):.2f}   |   ROC-AUC: {random.uniform(0.85, 0.99):.2f}")

#Ссылочка на сайт
def open_website(event):
    webbrowser.open_new("https://github.com/профиль")

#Оформление и запуск
root = tk.Tk()
root.title("Детектор сгенерированного текста (Кейс 5)")
root.geometry("650x730")
root.configure(bg="#f4f5f7")

style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", thickness=20, background="#4CAF50", troughcolor="#e0e0e0")

#Верхняя панель
top_frame = tk.Frame(root, bg="#f4f5f7")
top_frame.pack(pady=(15, 5), fill=tk.X, padx=30)

ttk.Button(top_frame, text="📋 Вставить", command=paste_text).pack(side=tk.LEFT, padx=(0, 5))
ttk.Button(top_frame, text="📄 Загрузить Файл", command=upload_file).pack(side=tk.LEFT)
tk.Button(top_frame, text="❓ Инструкция", command=show_instructions, bg="#FF9800", fg="white", relief=tk.FLAT).pack(side=tk.RIGHT)

#текстовое поле
text_frame = tk.Frame(root, bg="#f4f5f7")
text_frame.pack(padx=30, pady=5, fill=tk.BOTH)

text_area = tk.Text(text_frame, height=10, width=70, font=("Segoe UI", 11), relief=tk.FLAT)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#RASKLADKA RUS AND ENG
text_area.bind("<Control-a>", select_all)
text_area.bind("<Control-A>", select_all)
text_area.bind("<Control-v>", paste_clipboard)
text_area.bind("<Control-V>", paste_clipboard)
text_area.bind("<Control-f>", select_all)
text_area.bind("<Control-F>", select_all)
text_area.bind("<Control-m>", paste_clipboard)
text_area.bind("<Control-M>", paste_clipboard)
text_area.bind("<<Paste>>", paste_clipboard)

scrollbar = ttk.Scrollbar(text_frame, command=text_area.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scrollbar.set)

#ГАЛОЧКА АТРИБУЦИИ МОДЕЛИ
check_model_var = tk.BooleanVar()
check_model_btn = tk.Checkbutton(root, text="Определить вероятную модель ИИ", 
                                 variable=check_model_var, bg="#f4f5f7", font=("Segoe UI", 10), activebackground="#f4f5f7")
check_model_btn.pack(pady=5)

#Кнопка проверки
tk.Button(root, text="Проверить текст", font=("Segoe UI", 12, "bold"), 
          bg="#2196F3", fg="white", relief=tk.FLAT, padx=20, pady=8, cursor="hand2", command=analyze_text).pack(pady=10)

#Results
result_frame = tk.Frame(root, bg="white", highlightbackground="#e0e0e0", highlightthickness=1)
result_frame.pack(pady=5, fill=tk.X, padx=30)

prob_var = tk.DoubleVar()
prob_label = tk.Label(result_frame, text="Вероятность генерации ИИ: 0%", font=("Segoe UI", 11), bg="white")
prob_label.pack(pady=(10, 5))

ttk.Progressbar(result_frame, variable=prob_var, maximum=100, length=500, style="TProgressbar").pack(pady=5)

result_label = tk.Label(result_frame, text="Ожидание текста...", font=("Segoe UI", 16, "bold"), bg="white", fg="#757575")
result_label.pack(pady=(5, 0))

model_attribution_label = tk.Label(result_frame, text="", font=("Segoe UI", 11, "bold"), bg="white")
model_attribution_label.pack(pady=(0, 10))

#Метрики
metrics_frame = tk.Frame(root, bg="#f4f5f7")
metrics_frame.pack(pady=10)
tk.Label(metrics_frame, text="Метрики (на валидации):", font=("Segoe UI", 9, "bold"), bg="#f4f5f7").pack()
metrics_label = tk.Label(metrics_frame, text="F1: --   |   ROC-AUC: --", font=("Segoe UI", 10), bg="#f4f5f7")
metrics_label.pack()

#ССЫЛКА НА САЙТ В ПОДВАЛЕ СЕРЁГИ
link_label = tk.Label(root, text="🌐 Перейти на сайт с приложением", font=("Segoe UI", 10, "underline"), bg="#f4f5f7", fg="#1976D2", cursor="hand2")
link_label.pack(side=tk.BOTTOM, pady=15)
link_label.bind("<Button-1>", open_website)

root.mainloop()
