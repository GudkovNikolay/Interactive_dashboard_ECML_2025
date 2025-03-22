from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Paragraph, ColumnDataSource, Whisker, Range1d, Select, Label
import numpy as np
import pandas as pd
import pickle
from scipy.linalg import sqrtm
from fid import calculate_fid
from bokeh.driving import linear

from sharp_ratio import sharp_grid
    
# Функция для загрузки данных из CSV-файла
def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df

# Функция для расчета доверительного интервала (95%)
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data)
    confidence = 1.96 * std / np.sqrt(len(data))  # 95% доверительный интервал
    return mean, confidence

# Загружаем данные из файлов
real_data = load_data("data/real_data.csv")  # Загрузка реальных данных

# Создаем первый график (реальные данные)
plot1 = figure(title="Real Data Returns", x_axis_label='date', y_axis_label='return', width=600, height=300)
colors = ["blue", "green", "red", "purple", "orange", "grey", "pink", "brown"]
for i, col in enumerate(real_data.columns):
    y = real_data[col]
    plot1.line(x=np.arange(len(y)), y=y, line_width=2, color=colors[i])

# Добавляем пунктирную линию для разделения train/test
train_test_split = int(len(real_data) * 0.8)
plot1.vspan(x=train_test_split, line_width=2, line_dash="dashed", line_color="black")

# Добавляем надписи "Train" и "Test"
train_label = Label(x=train_test_split - 300, y=max(real_data.min()), text="Train", text_color="black")
test_label = Label(x=train_test_split + 100, y=max(real_data.min()), text="Test", text_color="black")
plot1.add_layout(train_label)
plot1.add_layout(test_label)

# Создаем третий график (синтетические данные)
plot3 = figure(title="Fake Data Returns", x_axis_label='date', y_axis_label='return', width=600, height=300)
fake_lines = []  # Для хранения линий синтетических данных

# Создаем heatmap (случайные данные)
heatmap1 = figure(title="Sharpe Ratio (real)", x_axis_label='n_start', y_axis_label='n_finish', width=300, height=300)
heatmap2 = figure(title="Sharpe Ratio (fake)", x_axis_label='n_start', y_axis_label='n_finish', width=300, height=300)

# Генерация случайных данных для heatmap
def generate_random_heatmap_data():
    return np.random.random((10, 10))

random_data1 = generate_random_heatmap_data()
random_data2 = generate_random_heatmap_data()

print(sharp_grid(real_data))
# Добавляем heatmap
heatmap1.image(image=[sharp_grid(real_data)], x=20, y=150, dw=20, dh=50, palette="Viridis256")

# heatmap1.image(image=sharp_grid(real_data), x=0, y=0, dw=10, dh=10, palette="Viridis256")
# heatmap2.image(image=[sharp_grid(real_data)], x=20, y=150, dw=20, dh=50, palette="Viridis256")

# Создаем выпадающий список для выбора модели
select_model = Select(title="Select Model:", options=["GARCH", "GAN"], value="GAN")

# Создаем кнопки
button_train = Button(label="Find Optimal Parameters with train", button_type="success")
button_fake = Button(label="Find Optimal Parameters with fake", button_type="success")
button_generate = Button(label="Generate Data (10 samples)", button_type="primary")

# Строка для отображения качества сгенерированных данных
generated_quality = Paragraph(text="Generated data quality (C-FID):", width=400)

# Переменные для хранения данных и индекса текущего файла
current_fake_data = None
current_index = 0
fid_scores = [0.0005446392709081585,
 0.0005723267577057196,
 0.0005941339247634619,
 0.0005673580326954015,
 0.0005606872544603801,
 0.0005638073175097487,
 0.0005057145926606131,
 0.00054834557407533,
 0.0005244690625734955,
 0.0004834589954559903]

# Функция для обновления графика синтетических данных
@linear()
def update_fake_data(step):
    global current_index, current_fake_data, fid_scores
    
    # Загружаем данные из файла
    current_fake_data = load_data(f"data/fake_{current_index}.csv")
    
    # Очищаем предыдущие линии
    for line in fake_lines:
        plot3.renderers.remove(line)
    fake_lines.clear()
    
    # Добавляем новые линии
    for i, col in enumerate(current_fake_data.columns):
        line = plot3.line(x=np.arange(len(current_fake_data[col])), y=current_fake_data[col], line_width=2, color=colors[i])
        fake_lines.append(line)
    
    # Обновляем индекс для следующего файла
    current_index = (current_index + 1) % 10
    
    # Рассчитываем C-FID
    if len(fid_scores) == 0:
        fid_score = [calculate_fid(real_data, load_data(f"data/fake_{j}.csv")) for j in range(10)]
    
    # Обновляем качество сгенерированных данных (среднее и стандартное отклонение)
    if len(fid_scores) == 10:
        mean_fid = np.mean(fid_scores)
        std_fid = np.std(fid_scores)
        generated_quality.text = f"Generated data quality (C-FID): Mean = {mean_fid}, Std = {std_fid}"

        # Обновляем heatmap
    sharp_fake = sharp_grid(current_fake_data)
    heatmap2.renderers = []  # Очищаем предыдущие данные
    heatmap2.image(image=[sharp_fake], x=20, y=150, dw=20, dh=50, palette="Viridis256")

# Функции-заглушки для кнопок
def find_optimal_parameters_train():
    print("Optimal parameters search started with train data...")  # Заглушка

def find_optimal_parameters_fake():
    print("Optimal parameters search started with fake data...")  # Заглушка

def generate_data():
    # Запускаем обновление графика с интервалом 2 секунды
    curdoc().add_periodic_callback(update_fake_data, 2000)

# Привязываем функции к кнопкам
button_train.on_click(find_optimal_parameters_train)
button_fake.on_click(find_optimal_parameters_fake)
button_generate.on_click(generate_data)

# Создаем поле для вывода "ожидаемой доходности"
expected_return = Paragraph(text="Optimal n_start: 0, optimal n_finish: 0, sharp ratio : 0.00", width=400)

# Создаем layout и добавляем его в документ
layout = column(
    row(select_model, button_generate, generated_quality),  # Верхняя строка с выпадающим списком и кнопкой генерации
    row(plot3, plot1),  # График с реальными данными
    row(heatmap1, heatmap2),  # Строка с двумя heatmap
    row(button_train, button_fake),  # Строка с кнопками
    expected_return
)
curdoc().add_root(layout)
