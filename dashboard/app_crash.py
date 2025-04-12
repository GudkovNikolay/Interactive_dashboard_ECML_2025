import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (RadioButtonGroup, Div, ColumnDataSource,
                          TextInput, LinearColorMapper,
                          BasicTicker, ColorBar)
from bokeh.plotting import figure
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.models.ranges import Range1d
from matplotlib import pyplot as plt

from library.constants import DEVICE
from library.dataset import get_pytorch_datataset
from library.gan import Generator
from library.gan_train_loop import load_gan
from library.generation import generate_fake_returns

# Constants
N_POINTS = 100
N_PROCESSES = 5
HEATMAP_SIZE = 10

# Generate fixed real Wiener processes
np.random.seed(42)
df_returns_real = get_pytorch_datataset()[0]
real_processes = np.array(df_returns_real.cumsum()).transpose()


# real_processes = np.cumsum(np.random.randn(N_PROCESSES, N_POINTS), axis=1)

# Generate different generated processes for each architecture
architectures = ['TCN', 'MLP', 'LSTM', 'GRU']
generated_processes = {
    arch: np.cumsum(np.random.randn(N_PROCESSES, N_POINTS), axis=1)
    for arch in architectures
}

generator = Generator(2).to(DEVICE)
load_gan('TCN', generator, epoch=800)

df_returns_fake = generate_fake_returns(generator, df_returns_real, seed=0)
x = df_returns_fake.index
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
#
# plt.subplot(2, 1, 1)

# df_returns_fake.cumsum().plot(ax=ax2)
# plt.title('Fake')
# plt.ylabel('Cumulative log-returns')
# plt.show()

generated_processes['TCN'] = np.array(df_returns_fake.cumsum()).transpose()

# Generate strategy returns for the bottom plot
train_returns = np.cumsum(np.random.randn(N_POINTS))
fake_returns = np.cumsum(np.random.randn(N_POINTS))
custom_returns = np.cumsum(np.random.randn(N_POINTS))

# Generate random C-FID values
cfid_values = {arch: np.random.uniform(1, 10) for arch in architectures}

# Generate heatmap data
heatmap_data_real = np.random.rand(HEATMAP_SIZE, HEATMAP_SIZE)
heatmap_data_generated = {
    arch: np.random.rand(HEATMAP_SIZE, HEATMAP_SIZE)
    for arch in architectures
}

# Generate random optimal parameters
optimal_params_train = {
    'n_start': np.random.randint(1, 5),
    'n_finish': np.random.randint(6, 10)
}

optimal_params_generated = {
    arch: {
        'n_start': np.random.randint(1, 5),
        'n_finish': np.random.randint(6, 10)
    }
    for arch in architectures
}

# Create data sources
real_source = ColumnDataSource(data={'x': x, **{f'y{i}': real_processes[i] for i in range(N_PROCESSES)}})
generated_source = ColumnDataSource(
    data={'x': x, **{f'y{i}': generated_processes['TCN'][i] for i in range(N_PROCESSES)}})

# For heatmaps
heatmap_x = np.tile(np.arange(HEATMAP_SIZE), HEATMAP_SIZE)
heatmap_y = np.repeat(np.arange(HEATMAP_SIZE), HEATMAP_SIZE)
heatmap_real_values = heatmap_data_real.flatten()
heatmap_generated_values = heatmap_data_generated['TCN'].flatten()

heatmap_real_source = ColumnDataSource(data={
    'x': heatmap_x,
    'y': heatmap_y,
    'values': heatmap_real_values
})

heatmap_generated_source = ColumnDataSource(data={
    'x': heatmap_x,
    'y': heatmap_y,
    'values': heatmap_generated_values
})

# For strategy returns plot
strategy_source = ColumnDataSource(data={
    'x': x,
    'train': train_returns,
    'fake': fake_returns,
    'custom': custom_returns
})

# Create widgets with correct styles attribute
architecture_selector = RadioButtonGroup(labels=architectures, active=0)
architecture_label = Div(text="<b>Architecture type</b>",
                         styles={'text-align': 'center'})

cfid_label = Div(text="<b>quality of generation (C-FID)</b>",
                 styles={'text-align': 'center'})
cfid_value = Div(text=f"{cfid_values['TCN']:.2f}",
                 styles={
                     'border': '1px solid gray',
                     'border-radius': '5px',
                     'padding': '5px',
                     'text-align': 'center',
                     'width': '100px',
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


# Create plots with fixed size and no stretching

def create_stock_plot(title, source):
    p = figure(
        title=title,
        width=400,
        height=300,
        tools="",
        toolbar_location=None,
        x_axis_type='datetime',  # Можно указать "datetime" если время
    )

    # Настройка внешнего вида оси X
    p.xaxis.axis_label_text_font_style = "normal"
    p.xaxis.axis_label_text_font_size = "12pt"

    # Добавление линий для каждого процесса
    for i in range(N_PROCESSES):
        p.line('x', f'y{i}', source=source, line_width=2)

    return p


real_plot = create_stock_plot("real stocks", real_source)
generated_plot = create_stock_plot("generated stocks", generated_source)


# Create heatmaps with fixed size
def create_heatmap(title, source):
    color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)

    p = figure(
        title=title,
        width=300,
        height=300,
        x_range=Range1d(0, HEATMAP_SIZE, bounds='auto'),
        y_range=Range1d(0, HEATMAP_SIZE, bounds='auto'),
        tools="",
        toolbar_location=None
    )

    p.rect(x='x', y='y', width=1, height=1, source=source,
           line_color=None, fill_color=transform('values', color_mapper))

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None)
    p.add_layout(color_bar, 'right')
    return p


heatmap_title = Div(text="<b>SR for momentum parameters</b>",
                    styles={'text-align': 'center'})
heatmap_real = create_heatmap("On real stocks", heatmap_real_source)
heatmap_generated = create_heatmap("On generated stocks", heatmap_generated_source)


# Create parameter displays with identical styling
def create_param_column(title, n_start, n_finish, is_custom=False):
    title_div = Div(text=f"<b>{title}</b>",
                    styles={'text-align': 'center', 'margin-bottom': '10px'})

    if is_custom:
        n_start_widget = TextInput(value="", title="n_start:", width=150)
        n_finish_widget = TextInput(value="", title="n_finish:", width=150)
    else:
        n_start_widget = create_param_display(f"n_start = {n_start}")
        n_finish_widget = create_param_display(f"n_finish = {n_finish}")

    return column(
        title_div,
        n_start_widget,
        n_finish_widget,
        align="center",
        styles={'margin': '0 30px'}  # Increased horizontal spacing
    )


# Create parameter columns
params_train = create_param_column(
    "optimal parameters with train",
    optimal_params_train['n_start'],
    optimal_params_train['n_finish']
)

params_generated = create_param_column(
    "optimal parameters with generated",
    optimal_params_generated['TCN']['n_start'],
    optimal_params_generated['TCN']['n_finish']
)

params_custom = create_param_column(
    "insert your parameters",
    "",
    "",
    is_custom=True
)

# Create strategy returns plot with fixed size
strategy_selector = RadioButtonGroup(labels=['train', 'fake', 'custom'], active=0)
strategy_plot = figure(
    title="Strategy returns",
    width=800,
    height=300,
    tools="",
    toolbar_location=None
)

# Initial plot with all lines (train highlighted)
strategy_plot.line('x', 'train', source=strategy_source, line_width=3, color='blue')
strategy_plot.line('x', 'fake', source=strategy_source, line_width=1, color='gray', alpha=0.2)
strategy_plot.line('x', 'custom', source=strategy_source, line_width=1, color='gray', alpha=0.2)


# Callback for architecture selection
def update_architecture(attr, old, new):
    selected_arch = architectures[architecture_selector.active]

    # Update generated processes
    new_data = {'x': x}
    for i in range(N_PROCESSES):
        new_data[f'y{i}'] = generated_processes[selected_arch][i]
    generated_source.data = new_data

    # Update C-FID value
    cfid_value.text = f"{cfid_values[selected_arch]:.2f}"

    # Update heatmap
    new_heatmap_data = heatmap_data_generated[selected_arch].flatten()
    heatmap_generated_source.data = {
        'x': heatmap_x,
        'y': heatmap_y,
        'values': new_heatmap_data
    }

    # Update optimal parameters
    params_generated.children[1].text = f"n_start = {optimal_params_generated[selected_arch]['n_start']}"
    params_generated.children[2].text = f"n_finish = {optimal_params_generated[selected_arch]['n_finish']}"


architecture_selector.on_change('active', update_architecture)


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
    column(cfid_label, cfid_value,
           styles={'margin': '0 auto'}),
    align="center",
    styles={'justify-content': 'center'}
)

# Top plots block with margin bottom
plots_block = column(
    row(
        real_plot,
        generated_plot,
        align="center",
        styles={'margin': '0 auto', 'justify-content': 'center'}
    ),
    styles={'margin-bottom': '40px'}  # Vertical space after this block
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

# Strategy plot block with margin top
strategy_block = column(
    strategy_selector,
    strategy_plot,
    align="center",
    styles={'margin': '40px auto 0'}  # Space above this block
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