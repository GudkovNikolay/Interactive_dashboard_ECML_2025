import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (RadioButtonGroup, Div, ColumnDataSource,
                          TextInput, LinearColorMapper, FixedTicker, BasicTicker,
                          BasicTicker, ColorBar, Select, Span, Label, Button, HoverTool, Legend)
from bokeh.plotting import figure
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.models.ranges import Range1d
from datetime import timedelta

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

# ========== Конфигурация ==========
N_START_VALUES = [20, 40, 60, 80, 100]
N_FINISH_VALUES = [150, 200, 250, 300, 350, 400]
GENERATIONS_AMOUNT = 100
GENERATIONS_COUNTER = 0
LINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
ARCHITECTURES = ['TCN', 'LSTM', 'GRU']

# ========== Заголовок и описание ==========
dashboard_title = Div(text="""
<h1 style='text-align: center; margin-bottom: 20px; font-size: 24px;'>
Interactive Dashboard for Momentum Strategy Evaluation with GANs
</h1>
<p style='text-align: center; margin-bottom: 30px; max-width: 800px; margin-left: auto; margin-right: auto;'>
<strong>This dashboard compares the quality of generated cumulative log 
returns for five stocks from the Moscow Exchange, evaluated using the 
C-FID metric.</strong> It also shows how momentum strategy performs with 
different parameters <strong>n finish</strong> and <strong>n start</strong>
 on train period of real data and on generated data, which allows to compare 
 optimal parameter sets, in terms of <strong>Sharpe Ratio</strong>. The GAN 
 architectures employed—<strong>TCN (Temporal Convolutional Network)</strong>, 
 <strong>LSTM (Long Short-Term Memory)</strong>, and <strong>GRU 
 (Gated Recurrent Unit)</strong>.
 </br> </br>

The momentum strategy is based on identifying trends in asset prices, 
buying assets that have shown upward momentum and selling those that have 
shown downward momentum. The parameters of strategy <strong>n start</strong> 
and <strong>n finish</strong> define the observation window for calculating 
the momentum signal. The graph in the bottom of the dashboard shows strategy 
performance on the test period, depending on the parameters set. There are 
three plots: strategy utilising parameters optimized with train period of 
real data (fixed), strategy utilising parameters optimized with generated 
data (depends on current generation), strategy utilising parameters optimized 
chosen by user. This graph allows to assess parameters impact on strategy performance.
</p>
""", styles={'margin': '20px 0 0 150px'})

# ========== Загрузка данных ==========
np.random.seed(42)
df_returns_real = get_pytorch_datataset()[0]
# print(f'legend = {df_returns_real.columns}')
N_POINTS = df_returns_real.shape[0]
N_PROCESSES = df_returns_real.shape[1]

real_processes = np.array(df_returns_real.cumsum()).transpose()

# Разделение на train/test
split_idx = int(len(df_returns_real) * 0.8)
train_data = df_returns_real.iloc[:split_idx]
test_data = df_returns_real.iloc[split_idx:]
split_date = df_returns_real.index[split_idx]

# ========== Генерация данных ==========

tcn_df_returns_fake = np.load('generated_returns/tcn_df_returns_fake.npy')
lstm_df_returns_fake = np.load('generated_returns/lstm_df_returns_fake.npy')
gru_df_returns_fake = np.load('generated_returns/gru_df_returns_fake.npy')

# Инициализация структур данных
generated_returns = {arch: [] for arch in ARCHITECTURES}
generated_processes = {arch: [] for arch in ARCHITECTURES}

generated_returns['TCN'] = [pd.DataFrame(tcn_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]
generated_returns['LSTM'] = [pd.DataFrame(lstm_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]
generated_returns['GRU'] = [pd.DataFrame(gru_df_returns_fake[i], index=df_returns_real.index) for i in range(GENERATIONS_AMOUNT)]

generated_processes['TCN'] = np.transpose(tcn_df_returns_fake.cumsum(axis=1), axes=(0, 2, 1))
generated_processes['LSTM'] = np.transpose(lstm_df_returns_fake.cumsum(axis=1), axes=(0, 2, 1))
generated_processes['GRU'] = np.transpose(gru_df_returns_fake.cumsum(axis=1), axes=(0, 2, 1))

# ========== Визуализация ==========
def create_stock_plot(title, source, split=False):
    p = figure(
        title=title,
        width=450,
        height=300,
        tools="",
        toolbar_location=None,
        x_axis_type='datetime'
    )
    p.title.text_font_size = '14pt'  # Правильный способ установки размера шрифта заголовка

    if split:
        split_line = Span(location=split_date, dimension='height',
                          line_color='red', line_width=1, line_dash='dashed')
        p.add_layout(split_line)

        train_label = Label(x=df_returns_real.index[int(split_idx * 0.4)],
                            y=0.9 * max(real_processes.flatten()),
                            text='train', text_color='red', text_font_size='12pt')
        test_label = Label(x=df_returns_real.index[split_idx + int((len(df_returns_real) - split_idx) * 0.3)],
                           y=0.9 * max(real_processes.flatten()),
                           text='test', text_color='red', text_font_size='12pt')
        p.add_layout(train_label)
        p.add_layout(test_label)
    test = []
    for i in range(N_PROCESSES):
        test.append(p.line('x', f'y{i}', source=source, line_width=2,
               color=LINE_COLORS[i % len(LINE_COLORS)],
               # legend_label=df_returns_real.columns[i]
                           )
                    )

    legend = Legend(items=[(legend, [val]) for legend, val in zip(df_returns_real.columns, test)], location="center", orientation = "horizontal", click_policy = "hide", label_text_font_size = "10pt")
    p.add_layout(legend, 'below')
    return p


data = np.array([[i * j for j in range(9)] for i in range(6)])




# def create_heatmap(title, source, color_mapper=None):
#     # Размеры данных
#     n_x = len(N_START_VALUES)
#     n_y = len(N_FINISH_VALUES)
#
#     # Размер ячейки (делаем квадратные ячейки)
#     cell_size = min(
#         (max(N_START_VALUES) - min(N_START_VALUES)) / n_x,
#         (max(N_FINISH_VALUES) - min(N_FINISH_VALUES)) / n_y
#     ) * 0.9  # 0.9 для небольшого отступа между ячейками
#
#     # Создаем цветовую карту если не передана
#     if color_mapper is None:
#         values = source.data['values']
#         color_mapper = LinearColorMapper(
#             palette=Viridis256,
#             low=min(values),
#             high=max(values)  # Исправлено: используем high вместо max
#         )
#
#     # Создаем фигуру
#     p = figure(
#         title=title.upper(),
#         x_range=(min(N_START_VALUES) - cell_size / 2, max(N_START_VALUES) + cell_size / 2),
#         y_range=(min(N_FINISH_VALUES) - cell_size / 2, max(N_FINISH_VALUES) + cell_size / 2),
#         tools="hover",
#         toolbar_location=None,
#         width=600,
#         height=400,
#         x_axis_label='n_start (days)',
#         y_axis_label='n_finish (days)'
#     )
#     p.title.text_font_size = '14pt'
#
#     # Рисуем квадратные ячейки без отступов
#     p.rect(
#         x='x', y='y',
#         width=cell_size,
#         height=cell_size,
#         source=source,
#         fill_color={'field': 'values', 'transform': color_mapper},
#         line_color=None
#     )
#
#     # Настройка осей
#     p.xaxis.ticker = N_START_VALUES
#     p.yaxis.ticker = N_FINISH_VALUES
#     p.grid.visible = False
#     p.outline_line_color = None
#
#     # Настройка подсказок
#     p.hover.tooltips = [
#         ("Parameters", "@x days / @y days"),
#         ("Sharpe Ratio", "@values{0.3f}")
#     ]
#
#     return p, color_mapper

# Определяем размеры данных
NUM_ROWS = len(N_START_VALUES)
NUM_COLS = len(N_FINISH_VALUES)

# Размеры
rows = len(N_START_VALUES)
cols = len(N_FINISH_VALUES)

# Подготовка данных для Bokeh
x = list(range(cols))  # Индексы для столбцов
y = list(range(rows))  # Индексы для строк
xx, yy = np.meshgrid(x, y)

# Сдвигаем координаты на половину единицы, чтобы центры ячеек совпадали с целыми числами
x_offset = xx.flatten() + 0.5
y_offset = yy.flatten() + 0.5

source = ColumnDataSource(data=dict(x=x_offset, y=y_offset, z=[d for d in data]))

def create_heatmap(data, x_ticks=N_FINISH_VALUES, y_ticks=N_START_VALUES, source=None, title="title", color_mapper=None):

    # Вычисляем width и height для сохранения квадратности
    base_size = 450  # Примерный базовый размер
    width = int(base_size * (cols / rows))  # Ширина пропорциональна кол-ву столбцов
    height = int(base_size * (cols / rows)**(-1))  # Высота пропорциональна кол-ву строк

    # Создание фигуры
    p = figure(width=width, height=height,
               x_range=(0, cols), y_range=(0, rows),
               x_axis_location="above",
               title=title,
               min_border=0,
               match_aspect=True,
               toolbar_location=None,
               tools='hover')

    # Рендеринг ячеек тепловой карты
    # Используем сдвинутые координаты x и y
    p.rect(x="x", y="y", width=1, height=1, source=source,
           fill_color={'field': 'values', 'transform': color_mapper},
           line_color=None)

    # Настройка осей
    # Создаем тикеры со смещением
    x_ticker = FixedTicker(ticks=[i + 0.5 for i in range(cols)])  # Сдвигаем тики в центр ячеек
    y_ticker = FixedTicker(ticks=[i + 0.5 for i in range(rows)])  # Сдвигаем тики в центр ячеек

    p.xaxis.ticker = x_ticker
    p.yaxis.ticker = y_ticker

    # Заменяем метки тиков на значения из массивов
    p.xaxis.major_label_overrides = {i + 0.5: str(val) for i, val in enumerate(x_ticks)}  # Преобразуем в строки
    p.yaxis.major_label_overrides = {i + 0.5: str(val) for i, val in enumerate(y_ticks)}  # Преобразуем в строки
    p.xaxis.axis_label = "n finish"
    p.yaxis.axis_label = "n start"
    p.xaxis.major_label_standoff = 0
    p.xaxis.major_tick_in = 0
    p.yaxis.major_label_standoff = 0
    p.yaxis.major_tick_in = 0

    # Настройка подсказок
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Sharpe ratio", "@values{0.3f}")
    ]
    hover.mode = 'mouse'
    # Создание цветовой шкалы (color bar)
    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                         label_standoff=12)

    # Добавление цветовой шкалы на график
    p.add_layout(color_bar, 'right')
    return p




heatmap_real_values = sharp_grid(df_returns_real).flatten()
heatmap_generated_values = sharp_grid(generated_returns['TCN'][GENERATIONS_COUNTER]).flatten()#heatmap_data_generated['TCN'].flatten()

heatmap_real_source = ColumnDataSource(data={
    'x': x_offset, #или xx.T.flatten() если нужно транспонировать
    'y': y_offset, #или yy.T.flatten() если нужно транспонировать
    'values': heatmap_real_values
})

heatmap_generated_source = ColumnDataSource(data={
    'x': x_offset,#или xx.T.flatten() если нужно транспонировать
    'y': y_offset, #или yy.T.flatten() если нужно транспонировать
    'values': heatmap_generated_values
})

all_values = np.concatenate([heatmap_real_source.data['values'],
                            heatmap_generated_source.data['values']])
common_color_mapper = LinearColorMapper(
    palette=Viridis256,
    low=min(all_values),
    high=max(all_values)
)

# Создаем хитмэпы с общей цветовой шкалой
heatmap_real = create_heatmap(data=heatmap_real_values, title="On Real Data (Train Period)", source=heatmap_real_source, color_mapper=common_color_mapper)
heatmap_generated = create_heatmap(data=heatmap_generated_values, title="On Generated Data", source=heatmap_generated_source, color_mapper=common_color_mapper)

# # Создание цветовой шкалы (color bar) - ОДИН для обоих графиков
# color_bar = ColorBar(color_mapper=common_color_mapper, ticker=FixedTicker(),
#                      label_standoff=12)
#
#
# # Добавление цветовой шкалы на график - добавляем к ОДНОМУ из графиков
# heatmap_real.add_layout(color_bar, 'right')  # Например, добавляем к heatmap_real
# heatmap_generated.add_layout(color_bar, 'right')

# ========== Источники данных ==========
real_source = ColumnDataSource(
    data={'x': df_returns_real.index, **{f'y{i}': real_processes[i] for i in range(N_PROCESSES)}})
generated_source = ColumnDataSource(
    data={'x': df_returns_real.index, **{f'y{i}': generated_processes['TCN'][0][i] for i in range(N_PROCESSES)}})

# ========== Виджеты ==========
architecture_selector = RadioButtonGroup(labels=ARCHITECTURES, active=0, width=300)
architecture_label = Div(text="<b>Gan Architecture</b>", styles={'text-align': 'center', 'font-size': '12pt'})

cfid_values = {
    'TCN': {'mean': 0.000224, 'std': 8.75e-06},
    'LSTM': {'mean': 0.001126, 'std': 2.13e-05},
    'GRU': {'mean': 0.000943, 'std': 2.84e-05}
}


def format_cfid(mean, std):
    if mean == 0 or std == 0:
        return f"{mean:.2e} ± {std:.2e}"
    order = -4
    cur_order = int(np.floor(np.log10(abs(mean))))
    scaled_std = std * (10 ** -order)
    scaled_mean = mean * (10 ** -order)
    return f"{scaled_mean:.2f}e{order} ± {scaled_std:.3f}e{order}"


cfid_label = Div(text="<b>Generation Quality (C-FID)</b>",
                 styles={'text-align': 'center', 'font-size': '12pt'})
cfid_value = Div(text=format_cfid(cfid_values['TCN']['mean'], cfid_values['TCN']['std']),
                 styles={
                     'border': '1px solid gray',
                     'border-radius': '5px',
                     'padding': '5px',
                     'text-align': 'center',
                     'width': '200px',
                     'margin': '0 auto',
                     'font-size': '11pt'
                 })

regenerate_button = Button(label="⟳ New Generation", button_type="default", width=150,
                           styles={'margin-left': '10px', 'margin-top': '37px', 'font-size': '12pt'})

# ========== Параметры стратегии ==========
def get_optimal_params(heatmap_source, N_START_VALUES, N_FINISH_VALUES):
  """
  Находит оптимальные параметры n_start и n_finish, соответствующие максимальному значению в плоском массиве heatmap_source.data['values'].

  Args:
    heatmap_source: Объект Bokeh ColumnDataSource, содержащий плоский массив значений в heatmap_source.data['values'].
    N_START_VALUES: Список возможных значений n_start.
    N_FINISH_VALUES: Список возможных значений n_finish.

  Returns:
    Словарь, содержащий оптимальные значения n_start и n_finish.
  """
  values = heatmap_source.data['values']
  max_index_flat = np.argmax(values)  # Индекс максимального значения в плоском массиве
  num_start_values = len(N_START_VALUES) # Число возможных значений n_finish
  num_finish_values = len(N_FINISH_VALUES)
  # Преобразуем плоский индекс в индексы по осям x (n_start) и y (n_finish)
  index_n_start = max_index_flat // num_finish_values  # Деление нацело для получения индекса n_start
  index_n_finish = max_index_flat % num_finish_values  # Остаток от деления для получения индекса n_finish
  optimal_params = {
      'n_start': N_START_VALUES[index_n_start],
      'n_finish': N_FINISH_VALUES[index_n_finish]
  }
  return optimal_params


# Применение функции к вашим данным:
optimal_params_train = get_optimal_params(heatmap_real_source, N_START_VALUES, N_FINISH_VALUES)
optimal_params_generated = get_optimal_params(heatmap_generated_source, N_START_VALUES, N_FINISH_VALUES)


n_start_select = Select(title="", value=str(N_START_VALUES[0]),
                        options=[str(x) for x in N_START_VALUES], width=150)
n_finish_select = Select(title="", value=str(N_FINISH_VALUES[0]),
                         options=[str(x) for x in N_FINISH_VALUES], width=150)


def create_param_block(title, n_start, n_finish, is_custom=False):
    title_div = Div(text=f"<b>{title}</b>",
                    styles={'text-align': 'center', 'margin-bottom': '10px', 'font-size': '12pt'})

    if is_custom:
        start_widget = n_start_select
        finish_widget = n_finish_select
    else:
        start_widget = Div(text=f"{n_start}", styles={'text-align': 'center', 'border': '1px solid gray',
                                                      'border-radius': '5px', 'padding': '5px', 'width': '150px',
                                                      'margin': '5px auto', 'font-size': '11pt'})
        finish_widget = Div(text=f"{n_finish}", styles={'text-align': 'center', 'border': '1px solid gray',
                                                        'border-radius': '5px', 'padding': '5px', 'width': '150px',
                                                        'margin': '5px auto', 'font-size': '11pt'})

    return column(
        title_div,
        Div(text="n start:", styles={'text-align': 'center', 'font-size': '11pt'}),
        start_widget,
        Div(text="n finish:", styles={'text-align': 'center', 'font-size': '11pt'}),
        finish_widget,
        align="center",
        styles={'margin': '0 20px'}
    )


params_train = create_param_block("Optimized on Real Data",
                                  optimal_params_train['n_start'],
                                  optimal_params_train['n_finish'])

params_generated = create_param_block("Optimized on Generated Data",
                                      optimal_params_generated['n_start'],
                                      optimal_params_generated['n_finish'])

params_custom = create_param_block("Custom Parameters", "", "", is_custom=True)

# ========== График стратегии ==========

train_returns = strategy_return(test_data, nf=optimal_params_train['n_finish'],
                                    ns=optimal_params_train['n_start']).cumsum()
fake_returns = strategy_return(test_data, nf=optimal_params_generated['n_finish'],
                               ns=optimal_params_generated['n_start']).cumsum()
custom_returns = strategy_return(test_data, nf=int(n_finish_select.value),
                                 ns=int(n_start_select.value)).cumsum()

strategy_source = ColumnDataSource(data={
    'x': test_data.index,
    'train': train_returns,
    'fake': fake_returns,
    'custom': custom_returns
})

strategy_selector = RadioButtonGroup(
    labels=['Real Params', 'Gen Params', 'Custom Params'],
    active=0,
    width=450
)

strategy_plot = figure(
    title="Strategy Returns (Test Period)",
    width=800,
    height=350,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    toolbar_location="right",
    x_axis_type='datetime',
    active_scroll="wheel_zoom"
)
strategy_plot.title.text_font_size = '14pt'  # Правильный способ установки размера шрифта заголовка

# hover = HoverTool(
#     tooltips=[("Date", "@x{%F}"), ("Return", "@$name{0.2f}")],
#     formatters={"@x": "datetime"},
#     mode='vline'
# )
# strategy_plot.add_tools(hover)

strategy_plot.line('x', 'train', source=strategy_source, line_width=2,
                   name="train", legend_label="Optimized on real data", color=LINE_COLORS[0])
strategy_plot.line('x', 'fake', source=strategy_source, line_width=2,
                   name="fake", legend_label="Optimized on generated data", color=LINE_COLORS[1])
strategy_plot.line('x', 'custom', source=strategy_source, line_width=2,
                   name="custom", legend_label="Custom parameters", color=LINE_COLORS[2])

strategy_plot.legend.location = "bottom_left"
strategy_plot.legend.click_policy = "hide"
strategy_plot.legend.label_text_font_size = "10pt"

# # ========== Кнопки масштабирования ==========
# zoom_buttons = RadioButtonGroup(
#     labels=["1M", "3M", "6M", "1Y", "ALL"],
#     active=4,
#     width=450,
#     styles={'margin': '10px auto'}
# )


# def zoom_callback(attr, old, new):
#     end_date = test_data.index[-1]
#     periods = [30, 90, 180, 365, None]
#
#     if periods[new] is None:
#         start_date = test_data.index[0]
#     else:
#         start_date = end_date - timedelta(days=periods[new])
#
#     strategy_plot.x_range.start = start_date
#     strategy_plot.x_range.end = end_date
#
#     # Автомасштабирование по Y
#     active_strategy = ['train', 'fake', 'custom'][strategy_selector.active]
#     visible_data = [y for x, y in zip(strategy_source.data['x'],
#                                       strategy_source.data[active_strategy])]
#     if visible_data:
#         y_padding = (max(visible_data) - min(visible_data)) * 0.1
#         strategy_plot.y_range.start = min(visible_data) - y_padding
#         strategy_plot.y_range.end = max(visible_data) + y_padding
#
# zoom_buttons.on_change('active', zoom_callback)

    # ========== Callback-функции ==========


def update_architecture(attr, old, new):
    selected_arch = ARCHITECTURES[new]

    # Обновление графиков
    new_data = {'x': df_returns_real.index}
    for i in range(N_PROCESSES):
        new_data[f'y{i}'] = generated_processes[selected_arch][GENERATIONS_COUNTER][i]
    generated_source.data = new_data

    # Обновление C-FID
    cfid_value.text = format_cfid(cfid_values[selected_arch]['mean'],
                                  cfid_values[selected_arch]['std'])

    # Обновление heatmap
    new_values = sharp_grid(generated_returns[selected_arch][GENERATIONS_COUNTER]).flatten()
    heatmap_generated_source.data['values'] = new_values

    # Обновление оптимальных параметров

    optimal_params_generated = get_optimal_params(heatmap_generated_source, N_START_VALUES, N_FINISH_VALUES)

    max_idx = np.argmax(new_values)
    new_n_start = optimal_params_generated["n_start"]
    new_n_finish = optimal_params_generated["n_finish"]

    params_generated.children[2].text = str(new_n_start)
    params_generated.children[4].text = str(new_n_finish)

    # Обновление стратегии
    new_fake_returns = strategy_return(test_data, nf=new_n_finish, ns=new_n_start).cumsum()
    strategy_source.data['fake'] = new_fake_returns


architecture_selector.on_change('active', update_architecture)


def regenerate_callback():
    global GENERATIONS_COUNTER
    GENERATIONS_COUNTER = (GENERATIONS_COUNTER + 1) % GENERATIONS_AMOUNT
    update_architecture(None, None, architecture_selector.active)


regenerate_button.on_click(regenerate_callback)


def update_custom_params(attr, old, new):
    try:
        n_start = int(n_start_select.value)
        n_finish = int(n_finish_select.value)
        custom_returns = strategy_return(test_data, nf=n_finish, ns=n_start).cumsum()
        strategy_source.data['custom'] = custom_returns
    except Exception as e:
        print(f"Error updating custom strategy: {e}")


n_start_select.on_change('value', update_custom_params)
n_finish_select.on_change('value', update_custom_params)


def update_strategy_display(attr, old, new):
    # Сброс всех линий
    for renderer in strategy_plot.renderers:
        if hasattr(renderer, 'glyph'):
            renderer.glyph.line_width = 1
            renderer.glyph.line_alpha = 0.2
            renderer.glyph.line_color = 'gray'

    # Выделение активной стратегии
    active_strategy = ['train', 'fake', 'custom'][new]
    colors = ['blue', 'green', 'red']

    strategy_plot.renderers[new].glyph.line_width = 3
    strategy_plot.renderers[new].glyph.line_alpha = 1
    strategy_plot.renderers[new].glyph.line_color = colors[new]

    # Автомасштабирование при переключении
    # zoom_callback(None, None, zoom_buttons.active)


strategy_selector.on_change('active', update_strategy_display)

# ========== Компоновка ==========
header = row(
    column(architecture_label, architecture_selector, styles={'margin': '0 0 0 50px'}),
    regenerate_button,
    column(cfid_label, cfid_value, styles={'margin': '0 auto'}),
    align="center",
    styles={'justify-content': 'center', 'margin': '20px 0'}
)

plots_block = column(
    row(
        create_stock_plot("Real Cumulative Log Returns", real_source, split=True),
        create_stock_plot("Generated Cumulative Log Returns", generated_source),
        align="center",
        styles={'justify-content': 'center'}
    ),
    styles={'margin': '20px 0 0 90px'}
)

heatmaps_block = column(
    Div(text="<b>Sharpe Ratio by Momentum Parameters</b>",
        styles={'text-align': 'center', 'font-size': '14pt', 'margin': '10px 0 0 380px'}),
    row(
        heatmap_real,
        heatmap_generated,
        align="center",
        styles={'justify-content': 'center', 'margin': '10px 0 0 10px'}
    ),
    styles={'margin': '30px 0'}
)

params_block = row(
    params_train,
    params_generated,
    params_custom,
    align="center",
    styles={'justify-content': 'center', 'margin': '30px 0'}
)

strategy_block = column(
    # strategy_selector,
    # zoom_buttons,
    strategy_plot,
    align="center",
    styles={'margin': '30px 0'}
)

dashboard = column(
    dashboard_title,
    header,
    plots_block,
    heatmaps_block,
    params_block,
    strategy_block,
    align="center",
    styles={'margin': '20px auto', 'max-width': '1300px'}
)

curdoc().add_root(dashboard)
curdoc().title = "Interactive Momentum Strategy Tool with GANs"