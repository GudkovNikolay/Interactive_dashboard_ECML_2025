import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (RadioButtonGroup, Div, ColumnDataSource,
                          TextInput, LinearColorMapper,
                          BasicTicker, ColorBar, Select, Span, Label, Button)

from bokeh.models import HoverTool

from bokeh.plotting import figure
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.models.ranges import Range1d
from matplotlib import pyplot as plt

from sharp_ratio import sharp_grid, strategy_return
from library.constants import DEVICE
from library.dataset import get_pytorch_datataset
from library.gan import Generator as TCN_Generator
from library.gan_LSTM import Generator as LSTM_Generator
from library.gan_GRU import Generator as GRU_Generator
from library.gan_train_loop import load_gan
from library.generation import generate_fake_returns as TCN_generate_fake_returns
from library.generation_LSTM import generate_fake_returns as LSTM_generate_fake_returns
from library.generation_GRU import generate_fake_returns as GRU_generate_fake_returns

from fid import calculate_fid

N_START_VALUES = [20, 40, 60, 80, 100]
N_FINISH_VALUES = [150, 200, 250, 300, 350, 400]

# Generate fixed real Wiener processes
np.random.seed(42)


# Кол-во генераций доходностей
GENERATIONS_AMOUNT = 100
GENERATIONS_COUNTER = 0

from library.constants import N_ASSETS

df_returns_real = get_pytorch_datataset()[0]#.cumsum()
real_processes = np.array(df_returns_real.cumsum()).transpose()

# Разделяем данные на train/test (80/20)
split_idx = int(len(df_returns_real) * 0.8)
train_data = df_returns_real.iloc[:split_idx]
test_data = df_returns_real.iloc[split_idx:]  # Сохраняем тестовые данные в отдельную переменную

# Получаем дату разделения
split_date = df_returns_real.index[split_idx]

# Constants
N_POINTS = df_returns_real.shape[0]
N_PROCESSES = df_returns_real.shape[1]
HEATMAP_SIZE = 10


# Generate fixed real Wiener processes
# np.random.seed(42)
# df_returns_real = get_pytorch_datataset()[0]
# real_processes = np.array(df_returns_real.cumsum()).transpose()


# real_processes = np.cumsum(np.random.randn(N_PROCESSES, N_POINTS), axis=1)

# Generate different generated processes for each architecture
architectures = ['TCN', 'LSTM', 'GRU']

generated_returns = {
    arch: np.random.randn(N_PROCESSES, N_POINTS)
    for arch in architectures
}
generated_processes = {
    arch: np.cumsum(np.random.randn(N_PROCESSES, N_POINTS), axis=1)
    for arch in architectures
}

tcn_df_returns_fake = np.load('dashboard/generated_returns/tcn_df_returns_fake.npy')
lstm_df_returns_fake = np.load('dashboard/generated_returns/lstm_df_returns_fake.npy')
gru_df_returns_fake = np.load('dashboard/generated_returns/gru_df_returns_fake.npy')

x = df_returns_real.index

# print('HERE')
# print(tcn_df_returns_fake[0].cumsum(axis=1).transpose().shape)
# print('HERE')

generated_processes['TCN'] = tcn_df_returns_fake[0].cumsum(axis=1).transpose()
generated_returns['TCN'] = [pd.DataFrame(tcn_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]

generated_processes['LSTM'] = lstm_df_returns_fake[0].cumsum(axis=1).transpose()
generated_returns['LSTM'] = [pd.DataFrame(lstm_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]

generated_processes['GRU'] = gru_df_returns_fake[0].cumsum(axis=1).transpose()
generated_returns['GRU'] = [pd.DataFrame(gru_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]

# Generate random C-FID values
# cfid_values = {arch: calculate_fid(df_returns_real, generated_returns[arch]) for arch in architectures}
# cfid_values = {arch: calculate_fid(df_returns_real, generated_returns[arch][GENERATIONS_COUNTER]) for arch in architectures}

cfid_values = {'TCN': {'mean': 0.000224809737867745, 'std': 8.750088205863424e-06},
 'LSTM': {'mean': 0.0011258203285543058, 'std': 2.1268350228962416e-05},
 'GRU': {'mean': 0.0009432847844564321, 'std': 2.8437548273114842e-05}}

# cfid_values['TCN'] = calculate_fid(df_returns_real, generated_returns['TCN'])

regenerate_button = Button(label="Regenerate",
                           button_type="default",
                           width=100,
                           styles={'margin-left': '20px'})




# Generate heatmap data
heatmap_data_real = np.random.rand(HEATMAP_SIZE, HEATMAP_SIZE)
heatmap_data_generated = {
    arch: np.random.rand(HEATMAP_SIZE, HEATMAP_SIZE)
    for arch in architectures
}

# Create data sources
real_source = ColumnDataSource(data={'x': x, **{f'y{i}': real_processes[i] for i in range(N_PROCESSES)}})
generated_source = ColumnDataSource(
    data={'x': x, **{f'y{i}': generated_processes['TCN'][i] for i in range(N_PROCESSES)}})

# For heatmaps
heatmap_x = np.tile(N_START_VALUES, len(N_FINISH_VALUES))
heatmap_y = np.repeat(N_FINISH_VALUES, len(N_START_VALUES))
heatmap_real_values = sharp_grid(df_returns_real).flatten()
heatmap_generated_values = sharp_grid(generated_returns['TCN'][GENERATIONS_COUNTER]).flatten()#heatmap_data_generated['TCN'].flatten()

# heatmap_real_source = ColumnDataSource(data={
#     'x': heatmap_x,
#     'y': heatmap_y,
#     'values': heatmap_real_values
# })
#
# heatmap_generated_source = ColumnDataSource(data={
#     'x': heatmap_x,
#     'y': heatmap_y,
#     'values': heatmap_generated_values
# })

N_START_VALUES = [20, 40, 60, 80, 100]
N_FINISH_VALUES = [150, 200, 250, 300, 350, 400]

# Подготовка данных
xx, yy = np.meshgrid(N_START_VALUES, N_FINISH_VALUES)

heatmap_real_source = ColumnDataSource(data={
    'x': xx.T.flatten(),
    'y': yy.T.flatten(),
    'values': heatmap_real_values.flatten()
})

heatmap_generated_source = ColumnDataSource(data={
    'x': xx.T.flatten(),
    'y': yy.T.flatten(),
    'values': heatmap_generated_values.flatten()
})

# Generate random optimal parameters
optimal_params_train = {
    'n_start': heatmap_real_source.data['x'][np.argmax(heatmap_real_source.data['values'])],
    'n_finish': heatmap_real_source.data['y'][np.argmax(heatmap_real_source.data['values'])]
}

optimal_params_generated = {
    'n_start': heatmap_generated_source.data['x'][np.argmax(heatmap_generated_source.data['values'])],
    'n_finish': heatmap_generated_source.data['y'][np.argmax(heatmap_generated_source.data['values'])]
}



# Create widgets with correct styles attribute
architecture_selector = RadioButtonGroup(labels=architectures, active=0)
architecture_label = Div(text="<b>Architecture type</b>",
                         styles={'text-align': 'center'})


def format_cfid(mean, std):
    if mean == 0 or std == 0:
        return f"{mean:.2e} ± {std:.2e}"

    # Определяем порядок среднего значения
    mean_order = int(np.floor(np.log10(abs(mean))))

    # Масштабируем стандартное отклонение к порядку среднего
    scaled_std = std * (10 ** -mean_order)

    # Форматируем с учетом нужного количества значащих цифр
    mean_str = f"{mean:.2e}"

    # Для стандартного отклонения:
    # - Если значение < 1.0, показываем 2 значащих цифры
    # - Иначе показываем 2 знака после запятой
    if scaled_std < 1.0:
        std_str = f"{scaled_std:.2g}".lstrip('0')
    else:
        std_str = f"{scaled_std:.2f}"

    # Собираем окончательную строку
    return f"{mean_str} ± {std_str}e{mean_order}"

cfid_label = Div(text="<b>quality of generation (C-FID)</b>",
                 styles={'text-align': 'center'})
cfid_value = Div(text=format_cfid(cfid_values['TCN']['mean'], cfid_values['TCN']['std']),
                 styles={
                     'border': '1px solid gray',
                     'border-radius': '5px',
                     'padding': '5px',
                     'text-align': 'center',
                     'width': '200px',
                     'margin': '0 auto'
                 })


# Create styled Div elements for parameter displays
def create_param_display(text):
    return Div(text=text,
               styles={
                   'border': '1px solid gray',
                   'border-radius': '5px',
                   'padding': '5px',
                   'text-align': 'center',
                   'width': '150px',
                   'margin': '5px auto'
               })

# Список цветов для линий (можно изменить по желанию)
LINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Модифицированная функция создания графиков
def create_stock_plot(title, source, split=False, legend=False):
    p = figure(
        title=title,
        width=400,
        height=300,
        tools="",
        toolbar_location=None,
        x_axis_type='datetime',
    )

    if split:
        # Добавляем вертикальную линию разделения
        split_line = Span(location=split_date,
                         dimension='height',
                         line_color='red',
                         line_width=1,
                         line_dash='dashed')
        p.add_layout(split_line)

        # Добавляем подписи train/test (сдвигаем test правее)
        train_label = Label(x=df_returns_real.index[int(split_idx * 0.4)],
                           y=0.9 * max(real_processes.flatten()),
                           text='train', text_color='red', text_font_size='10pt')
        test_label = Label(x=df_returns_real.index[split_idx + int((len(df_returns_real) - split_idx) * 0.3)],  # Сдвиг с 0.1 на 0.3
                          y=0.9 * max(real_processes.flatten()),
                          text='test', text_color='red', text_font_size='10pt')
        p.add_layout(train_label)
        p.add_layout(test_label)

    # Добавляем линии для каждого процесса с разными цветами


    if legend:
        for i in range(N_PROCESSES):
            p.line('x', f'y{i}', source=source,
                   line_width=2,
                   color=LINE_COLORS[i % len(LINE_COLORS)],
                   legend_label=df_returns_real.columns[i])
        # Настраиваем легенду
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"  # Клик скрывает/показывает линию
        p.legend.label_text_font_size = "8pt"
        p.legend.spacing = 1
        p.legend.padding = 5
        p.legend.margin = 5
    else:
        for i in range(N_PROCESSES):
            p.line('x', f'y{i}', source=source,
                   line_width=2,
                   color=LINE_COLORS[i % len(LINE_COLORS)])

    return p

# Создаем графики с новыми настройками
real_plot = create_stock_plot("real stocks", real_source, split=True, legend=True)
generated_plot = create_stock_plot("generated stocks", generated_source)



# Create heatmaps with fixed size
def create_heatmap(title, source):
    # Данные
    N_START_VALUES = [20, 40, 60, 80, 100]
    N_FINISH_VALUES = [150, 200, 250, 300, 350, 400]


    # Подготовка данных
    xx, yy = np.meshgrid(N_START_VALUES, N_FINISH_VALUES)

    # print(source.data)
    # print(source.data['values'])

    values = source.data['values']

    # Цветовая карта
    color_mapper = LinearColorMapper(
        palette=Viridis256,
        low=values.min(),
        high=values.max()
    )

    # Создание фигуры
    p = figure(
        title=title,
        x_range=Range1d(min(N_START_VALUES) - 10, max(N_START_VALUES) + 10),
        y_range=Range1d(min(N_FINISH_VALUES) - 25, max(N_FINISH_VALUES) + 25),
        tools="hover",  # ТОЛЬКО ПОДСКАЗКИ ПРИ НАВЕДЕНИИ
        toolbar_location=None,  # УБИРАЕМ ПАНЕЛЬ ИНСТРУМЕНТОВ
        width=600,
        height=400,
        x_axis_label='n_start',
        y_axis_label='n_finish'
    )

    # Прямоугольники хитмэпа
    p.rect(
        x='x', y='y',
        width=20,
        height=50,
        source=source,
        fill_color={'field': 'values', 'transform': color_mapper},
        line_color=None
    )

    # Цветовая шкала
    color_bar = ColorBar(
        color_mapper=color_mapper,
        width=20,
        location=(0, 0),
        title="Sharpe Ratio"
    )
    p.add_layout(color_bar, 'right')

    # Настройка подсказок
    p.hover.tooltips = [
        ("Start period", "@x days"),
        ("Finish period", "@y days"),
        ("Sharpe Ratio", "@values{0.3f}")
    ]

    # Чистый дизайн
    p.xaxis.ticker = N_START_VALUES
    p.yaxis.ticker = N_FINISH_VALUES
    p.grid.visible = False  # УБИРАЕМ СЕТКУ
    p.outline_line_color = None  # УБИРАЕМ РАМКУ
    return p


heatmap_title = Div(text="<b>SR for momentum parameters</b>",
                    styles={'text-align': 'center'})
heatmap_real = create_heatmap("On real stocks", heatmap_real_source)
heatmap_generated = create_heatmap("On generated stocks", heatmap_generated_source)


n_start_select = Select(
    title="n_start:",
    value=N_START_VALUES[0],
    options=[str(x) for x in N_START_VALUES],
    width=150
)

n_finish_select = Select(
    title="n_finish:",
    value=N_FINISH_VALUES[0],
    options=[str(x) for x in N_FINISH_VALUES],
    width=150
)

# Create parameter displays with identical styling
# Обновляем функцию создания колонки с параметрами
def create_param_column(title, n_start, n_finish, is_custom=False):
    title_div = Div(text=f"<b>{title}</b>",
                   styles={'text-align': 'center', 'margin-bottom': '10px'})

    if is_custom:
        # Используем Select вместо TextInput
        n_start_widget = n_start_select
        n_finish_widget = n_finish_select
    else:
        n_start_widget = create_param_display(f"n_start = {n_start}")
        n_finish_widget = create_param_display(f"n_finish = {n_finish}")

    return column(
        title_div,
        n_start_widget,
        n_finish_widget,
        align="center",
        styles={'margin': '0 30px'}
    )


# Стилизованный Div для отображения значений параметров
def create_param_value(value):
    return Div(
        text=f"<div style='text-align: center; border: 1px solid gray; border-radius: 5px; padding: 5px; width: 60px;'>{value}</div>",
        styles={'margin': '0 auto'})


# Общие стили для всех параметров
PARAM_BOX_STYLE = {
    'border': '1px solid gray',
    'border-radius': '5px',
    'padding': '5px',
    'text-align': 'center',
    'width': '100px',  # Фиксированная ширина для всех
    'margin': '5px auto'
}


# Функция создания единообразного блока параметров
def create_uniform_param_column(title, n_start=None, n_finish=None, is_select=False):
    title_div = Div(text=f"<b>{title}</b>",
                    styles={'text-align': 'center', 'margin-bottom': '10px'})

    # Блок для n_start
    start_label = Div(text="n_start:", styles={'text-align': 'center'})
    if is_select:
        start_widget = Select(options=[str(x) for x in N_START_VALUES],
                              value=str(N_START_VALUES[0]),
                              styles={'width': '100px'})  # Фиксируем ширину
    else:
        start_widget = Div(text=str(n_start), styles=PARAM_BOX_STYLE)

    # Блок для n_finish
    finish_label = Div(text="n_finish:", styles={'text-align': 'center'})
    if is_select:
        finish_widget = Select(options=[str(x) for x in N_FINISH_VALUES],
                               value=str(N_FINISH_VALUES[0]),
                               styles={'width': '100px'})  # Фиксируем ширину
    else:
        finish_widget = Div(text=str(n_finish), styles=PARAM_BOX_STYLE)

    return column(
        title_div,
        column(start_label, start_widget,
               finish_label, finish_widget,
               align="center"),
        align="center",
        styles={'margin': '0 30px'}
    )


# Создаем блоки параметров с единым стилем
params_train = create_uniform_param_column(
    "Optimal parameters (train)",
    optimal_params_train['n_start'],
    optimal_params_train['n_finish']
)

params_generated = create_uniform_param_column(
    "Optimal parameters (generated)",
    optimal_params_generated['n_start'],
    optimal_params_generated['n_finish']
)

# Блок с выпадающими списками (оставляем как было)
params_custom = create_param_column(
    "Select your parameters",
    "",
    "",
    is_custom=True
)

# # Generate strategy returns for the bottom plot
train_returns = strategy_return(test_data, nf=optimal_params_train['n_finish'], ns=optimal_params_train['n_start']).cumsum()
fake_returns = strategy_return(test_data, nf=optimal_params_generated['n_finish'], ns=optimal_params_generated['n_start']).cumsum()
custom_returns = strategy_return(test_data, nf=n_finish_select.value, ns=n_start_select.value).cumsum()

# For strategy returns plot
strategy_source = ColumnDataSource(data={
    'x': test_data.index,
    'train': train_returns,
    'fake': fake_returns,
    'custom': custom_returns
})

# Create strategy returns plot with fixed size
strategy_selector = RadioButtonGroup(labels=['train', 'fake', 'custom'], active=0)

# Модифицируем создание графика стратегий
strategy_plot = figure(
    title="Strategy returns",
    width=800,
    height=300,
    tools="pan,wheel_zoom,box_zoom,reset,save",  # Инструменты масштабирования
    toolbar_location="right",
    x_axis_type='datetime',
    active_scroll="wheel_zoom",
)

# Добавляем HoverTool с правильными форматировщиками
hover = HoverTool(
    tooltips=[
        ("Date", "@x{%F}"),
        ("Value", "@$name{0.2f}")
    ],
    formatters={
        "@x": "datetime",  # Формат даты
    },
    mode='vline'  # Вертикальная линия при наведении
)
strategy_plot.add_tools(hover)

# Добавляем линии стратегий с именами
strategy_plot.line('x', 'train', source=strategy_source,
                  line_width=3, color='blue', name="Train strategy")
strategy_plot.line('x', 'fake', source=strategy_source,
                  line_width=1, color='gray', alpha=0.2, name="Generated strategy")
strategy_plot.line('x', 'custom', source=strategy_source,
                  line_width=1, color='gray', alpha=0.2, name="Custom strategy")



# Настраиваем инструменты масштабирования
strategy_plot.toolbar.autohide = True  # Автоскрытие панели инструментов
strategy_plot.x_range.range_padding = 0.02  # Отступ по оси X
strategy_plot.y_range.range_padding = 0.1  # Отступ по оси Y

# Кнопки масштабирования (оставляем без изменений)
zoom_buttons = RadioButtonGroup(
    labels=["1M", "3M", "6M", "1Y", "All"],
    active=4,
    width=300,
    styles={'margin': '5px auto'}
)


# Callback для кнопок масштабирования (оставляем без изменений)
def zoom_callback(attr, old, new):
    from datetime import timedelta
    end_date = strategy_source.data['x'][-1]

    if zoom_buttons.active == 0:  # 1 месяц
        start_date = end_date - timedelta(days=30)
    elif zoom_buttons.active == 1:  # 3 месяца
        start_date = end_date - timedelta(days=90)
    elif zoom_buttons.active == 2:  # 6 месяцев
        start_date = end_date - timedelta(days=180)
    elif zoom_buttons.active == 3:  # 1 год
        start_date = end_date - timedelta(days=365)
    else:  # Все данные
        start_date = strategy_source.data['x'][0]

    strategy_plot.x_range.start = start_date
    strategy_plot.x_range.end = end_date

    # Автомасштабирование по Y
    visible_data = [y for x, y in zip(strategy_source.data['x'],
                                      strategy_source.data[strategy_selector.labels[strategy_selector.active]])
                    if start_date <= x <= end_date]
    if visible_data:
        y_padding = (max(visible_data) - min(visible_data)) * 0.1
        strategy_plot.y_range.start = min(visible_data) - y_padding
        strategy_plot.y_range.end = max(visible_data) + y_padding


zoom_buttons.on_change('active', zoom_callback)

# Callback for architecture selection
def update_architecture(attr, old, new):
    selected_arch = architectures[architecture_selector.active]

    # Update generated processes
    new_data = {'x': x}
    for i in range(N_PROCESSES):
        new_data[f'y{i}'] = generated_processes[selected_arch][i]
    generated_source.data = new_data

    # Update C-FID value
    # cfid_value.text = f"{cfid_values[selected_arch]['mean']:.2e} ({cfid_values[selected_arch]['std']:.2e})"
    mean = cfid_values[selected_arch]['mean']
    std = cfid_values[selected_arch]['std']
    cfid_value.text = format_cfid(mean, std)

    # Update heatmap
    new_heatmap_data = sharp_grid(generated_returns[selected_arch][GENERATIONS_COUNTER]).flatten()
    heatmap_generated_source.data['values'] = new_heatmap_data


    optimal_params_generated = {
        'n_start': heatmap_generated_source.data['x'][np.argmax(heatmap_generated_source.data['values'])],
        'n_finish': heatmap_generated_source.data['y'][np.argmax(heatmap_generated_source.data['values'])]
    }

    # Обновляем блок параметров generated
    params_generated.children[1].children[1].text = str(optimal_params_generated['n_start'])
    params_generated.children[1].children[3].text = str(optimal_params_generated['n_finish'])

    strategy_source.data['fake'] = strategy_return(test_data, nf=optimal_params_generated['n_finish'], ns=optimal_params_generated['n_start']).cumsum()
architecture_selector.on_change('active', update_architecture)

# 2. Создаем обработчик для кнопки
def regenerate_callback():
    # Получаем текущую выбранную архитектуру
    selected_arch = architectures[architecture_selector.active]
    global GENERATIONS_COUNTER
    global GENERATIONS_AMOUNT
    GENERATIONS_COUNTER = (GENERATIONS_COUNTER + 1) % GENERATIONS_AMOUNT

    generated_processes['TCN'] = np.array(tcn_df_returns_fake[GENERATIONS_COUNTER].cumsum()).transpose()
    # generated_returns['TCN'] = tcn_df_returns_fake

    generated_processes['LSTM'] = np.array(lstm_df_returns_fake[GENERATIONS_COUNTER].cumsum()).transpose()
    # generated_returns['LSTM'] = lstm_df_returns_fake

    generated_processes['GRU'] = np.array(gru_df_returns_fake[GENERATIONS_COUNTER].cumsum()).transpose()
    # generated_returns['GRU'] = gru_df_returns_fake
    # Генерируем новые данные с новым случайным seed
    # new_seed = np.random.randint(0, 10000)
    # new_returns = TCN_generate_fake_returns(tcn_generator, df_returns_real, seed=new_seed)
    #
    # # Обновляем данные для текущей архитектуры
    # generated_returns[selected_arch] = new_returns
    # generated_processes[selected_arch] = np.array(new_returns.cumsum()).transpose()

    new_returns = generated_returns[selected_arch][GENERATIONS_COUNTER]
    # Обновляем график
    new_data = {'x': x}
    for i in range(N_PROCESSES):
        new_data[f'y{i}'] = generated_processes[selected_arch][i]
    generated_source.data = new_data

    # Пересчитываем heatmap и оптимальные параметры
    heatmap_generated_values = sharp_grid(new_returns).flatten()
    heatmap_generated_source.data = {
        'x': heatmap_x,
        'y': heatmap_y,
        'values': heatmap_generated_values
    }

    # Находим новые оптимальные параметры
    values = np.array(heatmap_generated_values)
    max_idx = np.argmax(values)
    opt_n_start = heatmap_x[max_idx]
    opt_n_finish = heatmap_y[max_idx]

    # Обновляем отображаемые параметры
    optimal_params_generated[selected_arch] = {
        'n_start': int(opt_n_start),
        'n_finish': int(opt_n_finish)
    }
    params_generated.children[1].children[1].text = str(opt_n_start)
    params_generated.children[1].children[3].text = str(opt_n_finish)

    # Обновляем стратегию
    strategy_source.data['fake'] = strategy_return(test_data, nf=int(opt_n_start),
                                                     ns=int(opt_n_finish)).cumsum()




regenerate_button.on_click(regenerate_callback)

def update_custom(attr, old, new):
    # strategy_source.data['custom'] = strategy_return(test_data, nf=n_finish_select.value, ns=n_start_select.value).cumsum()

    try:
        strategy_source.data['custom'] = strategy_return(test_data, nf=int(n_finish_select.value),
                                                         ns=int(n_start_select.value)).cumsum()
    except Exception as e:
        print(f"Error updating strategy: {e}")
n_finish_select.on_change('value', update_custom)
n_start_select.on_change('value', update_custom)


# Callback for strategy selection
def update_strategy(attr, old, new):
    selected = strategy_selector.active

    # Reset all lines to gray and thin
    strategy_plot.renderers[0].glyph.line_width = 1
    strategy_plot.renderers[0].glyph.line_alpha = 0.2
    strategy_plot.renderers[0].glyph.line_color = 'gray'

    strategy_plot.renderers[1].glyph.line_width = 1
    strategy_plot.renderers[1].glyph.line_alpha = 0.2
    strategy_plot.renderers[1].glyph.line_color = 'gray'

    strategy_plot.renderers[2].glyph.line_width = 1
    strategy_plot.renderers[2].glyph.line_alpha = 0.2
    strategy_plot.renderers[2].glyph.line_color = 'gray'

    # Highlight selected line
    if selected == 0:  # train
        strategy_plot.renderers[0].glyph.line_width = 3
        strategy_plot.renderers[0].glyph.line_alpha = 1
        strategy_plot.renderers[0].glyph.line_color = 'blue'
    elif selected == 1:  # fake
        strategy_plot.renderers[1].glyph.line_width = 3
        strategy_plot.renderers[1].glyph.line_alpha = 1
        strategy_plot.renderers[1].glyph.line_color = 'green'
    else:  # custom
        strategy_plot.renderers[2].glyph.line_width = 3
        strategy_plot.renderers[2].glyph.line_alpha = 1
        strategy_plot.renderers[2].glyph.line_color = 'red'


strategy_selector.on_change('active', update_strategy)

# Layout with vertical spacing between blocks
header = row(
    column(architecture_label, architecture_selector,
           styles={'margin': '0 auto'}),
    regenerate_button,  # Добавляем кнопку справа
    column(cfid_label, cfid_value,
           styles={'margin': '0 auto'}),

    align="center",
    styles={'justify-content': 'center'}
)


# Добавляем легенду между графиками
legend_div = Div(text="<b>Stock colors legend:</b>",
                styles={'text-align': 'center', 'margin': '10px 0'})

# Модифицируем блок с графиками
plots_block = column(
    row(
        real_plot,
        generated_plot,
        align="center",
        styles={'margin': '0 auto', 'justify-content': 'center'}
    ),
    # legend_div,  # Добавляем легенду между графиками и хитмэпами
    styles={'margin-bottom': '20px'}  # Уменьшаем отступ
)

# Heatmaps block with margins
heatmaps_block = column(
    heatmap_title,
    row(
        heatmap_real,
        heatmap_generated,
        align="center",
        styles={'margin': '0 auto', 'justify-content': 'center'}
    ),
    align="center",
    styles={'margin': '40px auto'}  # Vertical space around this block
)

# Parameters block with increased horizontal spacing
params_block = row(
    params_train,
    params_generated,
    params_custom,
    align="center",
    styles={'justify-content': 'center', 'margin': '40px auto'}
)

# Модифицируем блок с графиком стратегий
strategy_block = column(
    strategy_selector,
    zoom_buttons,  # Добавляем кнопки масштабирования
    strategy_plot,
    align="center",
    styles={'margin': '40px auto 0'}
)

# Main dashboard layout
dashboard = column(
    header,
    plots_block,
    heatmaps_block,
    params_block,
    strategy_block,
    align="center",
    styles={'margin': '0 auto', 'max-width': '1200px'}
)

curdoc().add_root(dashboard)
curdoc().title = "Stocks Generation Dashboard"