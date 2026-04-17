# -*- coding: utf-8 -*-
"""
slide4_visualization.py — Визуализация: Объект и база исследования
Стиль: академический, минималистичный, для презентации
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patches as mpatches

# ==================== НАСТРОЙКИ СТИЛЯ ====================
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Arial']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Цветовая палитра
COLOR_PRIMARY   = '#2E86C1'   # синий
COLOR_SECONDARY = '#28B463'   # зелёный
COLOR_ACCENT    = '#8E44AD'   # фиолетовый
COLOR_BG        = '#F8F9FA'   # светло-серый фон
COLOR_TEXT      = '#2C3E50'   # тёмный текст

# ==================== ЗАГОЛОВОК СЛАЙДА ====================
ax.text(5, 5.6, "Слайд 4. Объект и база исследования",
        ha='center', fontsize=14, fontweight='bold', color=COLOR_TEXT)

# ==================== ЛЕВАЯ КОЛОНКА: Объект и Предмет ====================
# Блок "Объект"
obj_box = FancyBboxPatch((0.4, 3.8), 4.0, 1.3,
                         boxstyle="round,pad=0.3",
                         fc=COLOR_BG, ec=COLOR_PRIMARY, lw=1.8, zorder=3)
ax.add_patch(obj_box)
ax.text(2.4, 4.65, "🏢 Объект", ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=COLOR_PRIMARY, zorder=4)
ax.text(2.4, 4.25, "ООО «Промтехсервис»", ha='center', va='top',
        fontsize=10, color=COLOR_TEXT, zorder=4)
ax.text(2.4, 3.95, "строительство жилых и нежилых зданий",
        ha='center', va='top', fontsize=9, style='italic', color='#555', zorder=4)

# Блок "Предмет"
subj_box = FancyBboxPatch((0.4, 2.2), 4.0, 1.3,
                          boxstyle="round,pad=0.3",
                          fc=COLOR_BG, ec=COLOR_SECONDARY, lw=1.8, zorder=3)
ax.add_patch(subj_box)
ax.text(2.4, 3.05, "🔍 Предмет анализа", ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=COLOR_SECONDARY, zorder=4)
ax.text(2.4, 2.55, "Бизнес-процесс поиска нормативных обоснований",
        ha='center', va='center', fontsize=9.5, color=COLOR_TEXT, zorder=4, linespacing=1.1)
ax.text(2.4, 2.32, "инженерами-сметчиками", ha='center', va='top',
        fontsize=9, style='italic', color='#555', zorder=4)

# ==================== ПРАВАЯ КОЛОНКА: Информационная база ====================
ax.text(7.0, 4.9, "📚 Информационная база", ha='center',
        fontsize=12, fontweight='bold', color=COLOR_ACCENT, zorder=4)

# Иконка "База данных" (цилиндр)
db_x, db_y = 7.0, 3.8
# Верхний эллипс
ax.add_patch(Circle((db_x, db_y + 0.25), 0.55, fc=COLOR_PRIMARY, ec='black', lw=1.2, zorder=3))
# Бока цилиндра
ax.plot([db_x - 0.55, db_x - 0.55], [db_y + 0.25, db_y - 0.35], color='black', lw=1.2, zorder=3)
ax.plot([db_x + 0.55, db_x + 0.55], [db_y + 0.25, db_y - 0.35], color='black', lw=1.2, zorder=3)
# Нижний эллипс (дуга)
theta = np.linspace(np.pi, 2*np.pi, 50)
ax.plot(db_x + 0.55 * np.cos(theta), db_y - 0.35 + 0.15 * np.sin(theta),
        color='black', lw=1.2, zorder=3)
# Полоски на цилиндре
for dy in [0.05, -0.15]:
    ax.plot([db_x - 0.5, db_x + 0.5], [db_y + dy, db_y + dy],
            color='white', lw=2.5, alpha=0.7, zorder=4)

# Список источников
sources = [
    ("📜", "Нормативы Минстроя", "(ФСНБ, ГЭСН, ФЕР)"),
    ("❓", "База вопросов-ответов ГГЭ", "(≈2 000 записей)"),
    ("📊", "Внутренние данные предприятия", "(2022–2024 гг.)"),
]
for i, (icon, title, subtitle) in enumerate(sources):
    y_pos = 3.0 - i * 0.75
    # Фон строки
    ax.add_patch(FancyBboxPatch((5.4, y_pos - 0.22), 3.2, 0.45,
                                 boxstyle="round,pad=0.2",
                                 fc='white', ec='#ccc', lw=0.8, zorder=2))
    ax.text(5.6, y_pos + 0.08, f"{icon} {title}", ha='left', va='center',
            fontsize=9.5, fontweight='bold', color=COLOR_TEXT, zorder=3)
    ax.text(8.5, y_pos - 0.05, subtitle, ha='right', va='center',
            fontsize=8.5, color='#666', style='italic', zorder=3)

# ==================== БЕЙДЖИ ФОРМАТОВ (внизу справа) ====================
formats = [('PDF', '#E74C3C'), ('XML', '#27AE60'), ('JSON', '#F39C12')]
for i, (fmt, color) in enumerate(formats):
    x = 6.0 + i * 1.1
    badge = FancyBboxPatch((x - 0.45, 0.45), 0.9, 0.5,
                           boxstyle="round,pad=0.15",
                           fc=color, ec='white', lw=1.5, zorder=5)
    ax.add_patch(badge)
    ax.text(x, 0.7, fmt, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)

# Подпись под бейджами
ax.text(7.0, 0.15, "форматы источников", ha='center', va='top',
        fontsize=8.5, color='#777', style='italic', zorder=4)

# ==================== ДЕКОРАТИВНАЯ ЛИНИЯ-РАЗДЕЛИТЕЛЬ ====================
ax.plot([4.7, 4.7], [0.8, 5.2], color='#ddd', lw=1.2, linestyle=':', zorder=1)

# ==================== СОХРАНЕНИЕ ====================
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
plt.savefig('slide4_object_base.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('slide4_object_base.pdf', format='pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Слайд 4 сохранён: slide4_object_base.png / .pdf")
print("   • Объект и предмет — слева")
print("   • Иконка БД + источники — справа")
print("   • Бейджи PDF/XML/JSON — внизу как иллюстрация форматов")