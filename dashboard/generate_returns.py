import numpy as np

from library.constants import DEVICE
from library.dataset import get_pytorch_datataset
from library.gan import Generator as TCN_Generator
from library.gan_LSTM import Generator as LSTM_Generator
from library.gan_GRU import Generator as GRU_Generator
from library.gan_train_loop import load_gan
from library.generation import generate_fake_returns as TCN_generate_fake_returns
from library.generation_LSTM import generate_fake_returns as LSTM_generate_fake_returns
from library.generation_GRU import generate_fake_returns as GRU_generate_fake_returns

GENERATIONS_AMOUNT = 100

df_returns_real = get_pytorch_datataset()[0]

tcn_generator = TCN_Generator(2).to(DEVICE)
load_gan('TCN', tcn_generator, epoch=800)
tcn_df_returns_fake = [TCN_generate_fake_returns(tcn_generator, df_returns_real, seed=i) for i in range(GENERATIONS_AMOUNT)]
np.save('generated_returns/tcn_df_returns_fake.npy', np.array(tcn_df_returns_fake))

lstm_generator = LSTM_Generator().to(DEVICE)
load_gan('LSTM', lstm_generator, epoch=800)
lstm_df_returns_fake = [LSTM_generate_fake_returns(lstm_generator, df_returns_real, seed=i) for i in range(GENERATIONS_AMOUNT)]
np.save('generated_returns/lstm_df_returns_fake.npy', np.array(lstm_df_returns_fake))


gru_generator = GRU_Generator().to(DEVICE)
load_gan('GRU', gru_generator, epoch=800)
gru_df_returns_fake = [GRU_generate_fake_returns(gru_generator, df_returns_real, seed=i) for i in range(GENERATIONS_AMOUNT)]
np.save('generated_returns/gru_df_returns_fake.npy', np.array(gru_df_returns_fake))
