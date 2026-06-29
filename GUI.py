import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import webbrowser

def paste_text():
    """Функция вставки текста из буфера обмена"""
    try:
        clipboard_text = root.clipboard_get()
        text_area.delete("1.0", tk.END) # Очищаем текущее поле
        text_area.insert(tk.END, clipboard_text) # Вставляем текст
    except tk.TclError:
        messagebox.showwarning("Внимание", "Буфер обмена пуст или содержит не текстовые данные.")

def upload_file():
    """Функция загрузки текста из .txt файла"""
    filepath = filedialog.askopenfilename(
        title="Выберите текстовый файл",
        filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
    )
    if filepath:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                text_area.delete("1.0", tk.END)
                text_area.insert(tk.END, content)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{e}")

def analyze_text():
    """Функция анализа текста(проба)"""
    input_text = text_area.get("1.0", tk.END).strip()
    
    if not input_text:
        result_label.config(text="Добавьте текст для анализа.", fg="#d32f2f")
        return

    # Жалкая имитация работы нейросети(Разве может робот написать симфонию? А взять чистый холст и превратить его в шедевр?) 
    ai_probability = random.uniform(0, 100)
    
    #Обновляем Progress Bar
    prob_var.set(ai_probability)
    prob_label.config(text=f"Вероятность генерации ИИ: {ai_probability:.1f}%")
    
    #Бинарная классификация (Порог отсечения - 50%)
    if ai_probability >= 50.0:
        verdict = "Сгенерировано ИИ"
        color = "#d32f2f" # Красный
    else:
        verdict = "Написано человеком"
        color = "#388e3c" # Зеленый
        
    result_label.config(text=f"Вердикт: {verdict}", fg=color)
    
    #Случайные метрики (в проекте будем выводить метрики валидации модели)
    f1_score = random.uniform(0.80, 0.99)
    roc_auc = random.uniform(0.85, 0.99)
    metrics_label.config(text=f"F1: {f1_score:.2f}   |   ROC-AUC: {roc_auc:.2f}")

def open_website(event):
    """Открытие ссылки на модель в браузере"""
    webbrowser.open_new("https://сайт с моделью.пупупуу")

#ОФОРМЛЕНИЕ + ЗАПУСК
root = tk.Tk()
root.title("Детектор сгенерированного текста")
root.geometry("650x650")
root.configure(bg="#f4f5f7")
root.resizable(False, False)

# Настройка стилей
style = ttk.Style()
style.theme_use('clam')
#Progressbar 
style.configure("TProgressbar", thickness=20, background="#4CAF50", troughcolor="#e0e0e0", bordercolor="#f4f5f7", lightcolor="#4CAF50", darkcolor="#4CAF50")
#кнопки ttk
style.configure("TButton", font=("Segoe UI", 10), padding=5)

#ВЕРХНЯЯ ПАНЕЛЬ С КНОПКАМИ(Вставить / Загрузить)
top_frame = tk.Frame(root, bg="#f4f5f7")
top_frame.pack(pady=(20, 10), fill=tk.X, padx=30)

# Используем юникод-иконки
btn_paste = ttk.Button(top_frame, text="📋 Вставить текст", command=paste_text)
btn_paste.pack(side=tk.LEFT, padx=(0, 10))

btn_upload = ttk.Button(top_frame, text="📄 Загрузить файл", command=upload_file)
btn_upload.pack(side=tk.LEFT)

#ТЕКСТОВОЕ ПОЛЕ
text_frame = tk.Frame(root, bg="#f4f5f7")
text_frame.pack(padx=30, pady=5, fill=tk.BOTH)

text_area = tk.Text(text_frame, height=12, width=70, font=("Segoe UI", 11), relief=tk.FLAT, bd=1)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
text_area.config(highlightbackground="#cccccc", highlightcolor="#2196F3", highlightthickness=1) # Рамка при фокусе

#Скроллбар для текста
scrollbar = ttk.Scrollbar(text_frame, command=text_area.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scrollbar.set)

#ГЛАВНАЯ КНОПКА ПРОВЕРКИ
check_btn = tk.Button(root, text="Проверить текст", font=("Segoe UI", 12, "bold"), 
                      bg="#2196F3", fg="white", activebackground="#1976D2", activeforeground="white",
                      relief=tk.FLAT, padx=20, pady=8, cursor="hand2", command=analyze_text)
check_btn.pack(pady=15)

#БЛОК РЕЗУЛЬТАТОВ И ПРОГРЕСС БАРА
result_frame = tk.Frame(root, bg="white", highlightbackground="#e0e0e0", highlightthickness=1)
result_frame.pack(pady=10, fill=tk.X, padx=30)

prob_var = tk.DoubleVar()
prob_label = tk.Label(result_frame, text="Вероятность генерации ИИ: 0%", font=("Segoe UI", 11), bg="white")
prob_label.pack(pady=(10, 5))

# Сам прогресс-бар
prob_bar = ttk.Progressbar(result_frame, variable=prob_var, maximum=100, length=500, style="TProgressbar")
prob_bar.pack(pady=5)

# Вывод бинарного вердикта
result_label = tk.Label(result_frame, text="Ожидание текста...", font=("Segoe UI", 15, "bold"), bg="white", fg="#757575")
result_label.pack(pady=(5, 10))

#БЛОК МЕТРИК
metrics_frame = tk.Frame(root, bg="#f4f5f7")
metrics_frame.pack(pady=5)
tk.Label(metrics_frame, text="Показатели качества модели (на валидации):", font=("Segoe UI", 9, "bold"), bg="#f4f5f7", fg="#555").pack()
metrics_label = tk.Label(metrics_frame, text="F1: --   |   ROC-AUC: --", font=("Segoe UI", 10), bg="#f4f5f7", fg="#555")
metrics_label.pack()

#ССЫЛКА НА САЙТ В ПОДВАЛЕ СЕРЁГИ
link_frame = tk.Frame(root, bg="#f4f5f7")
link_frame.pack(side=tk.BOTTOM, pady=20)

link_label = tk.Label(link_frame, 
                      text="🌐 Перейти на сайт с приложением", 
                      font=("Segoe UI", 10, "underline"), 
                      bg="#f4f5f7", fg="#1976D2", 
                      cursor="hand2")
link_label.pack()
link_label.bind("<Button-1>", open_website)

root.mainloop()