import streamlit as st
import numpy as np
from scipy.spatial import distance
import ot
import matplotlib.pyplot as plt
import ot.plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
plt.rcParams["font.size"] = 10
img_height = 10
figsize = (img_height, img_height)
marker_size = 100
st.set_page_config(layout = 'wide')


st.title('点群の場合の最適輸送の可視化')

st.header('入力の設定')

n = st.number_input('$n$の設定(1 ~ 100)', min_value = 1, max_value = 100, value = 10)
m = st.number_input('$m$の設定(1 ~ 100)', min_value = 1, max_value = 100, value = 10)

method = st.radio('確率ベクトル $a\in\Sigma_n, b\in\Sigma_m$ の生成方法の設定', ('一様', 'ランダム'))
if method == 'ランダム':
  SEED = st.slider('確率ベクトルの生成用の乱数のシード値の設定', min_value = 0, max_value = 500, step = 1, value = 314)
  np.random.seed(SEED)
  a = np.random.rand(n)
  b = np.random.rand(m)
else:
  a = np.ones(n)
  b = np.ones(m)
st.write('確率ベクトル$a, b$は以下のように生成された．')
a /= np.sum(a)
b /= np.sum(b)
st.write(f'a = {a}')
st.write(f'b = {b}')
# for plotting
mx_a = np.max(a)
mx_b = np.max(b)

SEED_ = st.slider('$x,y$の座標の生成用の乱数のシード値の設定', min_value = 0, max_value = 500, step = 1, value = 314)
np.random.seed(SEED_)
x = np.random.rand(n, 2) * 10
y = np.random.rand(m, 2) * 10

metric = st.radio('コスト関数の設定', ('ユークリッド距離', '平方ユークリッド距離', 'マンハッタン距離', 'ミンコフスキー距離(pは後で決める)'))
if metric == 'ミンコフスキー距離(pは後で決める)':
  p = st.slider('$p$の設定($p<1$のとき，ミンコフスキー距離は距離の公理を満たさない．$p = 1, 2$のとき，それぞれマンハッタン距離とユークリッド距離に一致する．)', min_value = 0.1, max_value = 10.0, value = 2.0)
  dist = 'minkowski'
else:
  if metric == 'ユークリッド距離':
    dist = 'euclidean'
  elif metric == '平方ユークリッド距離':
    dist = 'sqeuclidean'
  else:
    dist = 'cityblock'
st.write('ユークリッド距離 : $\|x - y\|_2$')
st.write('平方ユークリッド距離 : $\|x - y\|_2^2$')
st.write('マンハッタン距離 : $\sum_i\|x_i - y_i\|$')
st.write('ミンコフスキー距離 : $\|x - y\|_p, where\:\|x\|_p = (\sum_i|x_i|^p)^{1/p}$')

col1, col2 = st.columns(2)

with col1:
  st.header('点群の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  ax.scatter(x[:, 0], x[:, 1], marker = 'o', label = 'Source samples', alpha = a / mx_a, s = marker_size, c = 'r')
  ax.scatter(y[:, 0], y[:, 1], marker = 'x', label = 'Target samples', alpha = b / mx_b, s = marker_size, c = 'b')
  for i in range(n):
    ax.text(x[i, 0] - 0.1, x[i, 1] + 0.1, '$x^{(' + str(i) + ')}$', c = 'r')
  for i in range(m):
    ax.text(y[i, 0] - 0.1, y[i, 1] + 0.1, '$y^{(' + str(i) + ')}$', c = 'b')
  ax.legend()
  ax.set_title('Source and target distributions')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)

with col2:
  st.header('コスト行列の可視化')
  C = distance.cdist(x, y, metric = dist)
  fig, ax = plt.subplots(figsize = figsize)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', '5%', pad = '3%')
  im = ax.imshow(C, cmap = plt.cm.binary)
  fig.colorbar(im, cax = cax)
  ax.set_title('Cost matrix')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)



P_star = ot.emd(a, b, C)
col1, col2 = st.columns(2)

with col1:
  st.header('最適輸送の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  ot.plot.plot2D_samples_mat(x, y, P_star, color = [.5, .5, 1])
  ax.scatter(x[:, 0], x[:, 1], marker = 'o', label = 'Source samples', alpha = a / mx_a, s = marker_size, c = 'r')
  ax.scatter(y[:, 0], y[:, 1], marker = 'x', label = 'Target samples', alpha = b / mx_b, s = marker_size, c = 'b')
  for i in range(n):
    ax.text(x[i, 0] - 0.1, x[i, 1] + 0.1, '$x^{(' + str(i) + ')}$', c = 'r')
  for i in range(m):
    ax.text(y[i, 0] - 0.1, y[i, 1] + 0.1, '$y^{(' + str(i) + ')}$', c = 'b')
  ax.legend()
  ax.set_title('Optimal Transport')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)

with col2:
  st.header('最適輸送行列の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', '5%', pad = '3%')
  im = ax.imshow(P_star, cmap = plt.cm.binary)
  fig.colorbar(im, cax = cax)
  ax.set_title('OT matrix')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)

with st.expander('参考：'):
  st.write('最適輸送本のサポートページ[(https://github.com/joisino/otbook)](https://github.com/joisino/otbook)')
  st.write('最適輸送のpythonのライブラリーであるPOTのexample code[(https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html)](https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html)')

with st.expander('このアプリのプログラムコード：'):
  code = '''
import streamlit as st
import numpy as np
from scipy.spatial import distance
import ot
import matplotlib.pyplot as plt
import ot.plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
plt.rcParams["font.size"] = 10
img_height = 10
figsize = (img_height, img_height)
marker_size = 100
st.set_page_config(layout = 'wide')


st.title('点群の場合の最適輸送の可視化')

st.header('入力の設定')

n = st.number_input('$n$の設定(1 ~ 100)', min_value = 1, max_value = 100, value = 10)
m = st.number_input('$m$の設定(1 ~ 100)', min_value = 1, max_value = 100, value = 10)

method = st.radio('確率ベクトル $a\in\Sigma_n, b\in\Sigma_m$ の生成方法の設定', ('一様', 'ランダム'))
if method == 'ランダム':
  SEED = st.slider('確率ベクトルの生成用の乱数のシード値の設定', min_value = 0, max_value = 500, step = 1, value = 314)
  np.random.seed(SEED)
  a = np.random.rand(n)
  b = np.random.rand(m)
else:
  a = np.ones(n)
  b = np.ones(m)
st.write('確率ベクトル$a, b$は以下のように生成された．')
a /= np.sum(a)
b /= np.sum(b)
st.write(f'a = {a}')
st.write(f'b = {b}')
# for plotting
mx_a = np.max(a)
mx_b = np.max(b)

SEED_ = st.slider('$x,y$の座標の生成用の乱数のシード値の設定', min_value = 0, max_value = 500, step = 1, value = 314)
np.random.seed(SEED_)
x = np.random.rand(n, 2) * 10
y = np.random.rand(m, 2) * 10

metric = st.radio('コスト関数の設定', ('ユークリッド距離', '平方ユークリッド距離', 'マンハッタン距離', 'ミンコフスキー距離(pは後で決める)'))
if metric == 'ミンコフスキー距離(pは後で決める)':
  p = st.slider('$p$の設定($p<1$のとき，ミンコフスキー距離は距離の公理を満たさない．$p = 1, 2$のとき，それぞれマンハッタン距離とユークリッド距離に一致する．)', min_value = 0.1, max_value = 10.0, value = 2.0)
  dist = 'minkowski'
else:
  if metric == 'ユークリッド距離':
    dist = 'euclidean'
  elif metric == '平方ユークリッド距離':
    dist = 'sqeuclidean'
  else:
    dist = 'cityblock'
st.write('ユークリッド距離 : $\|x - y\|_2$')
st.write('平方ユークリッド距離 : $\|x - y\|_2^2$')
st.write('マンハッタン距離 : $\sum_i\|x_i - y_i\|$')
st.write('ミンコフスキー距離 : $\|x - y\|_p, where\:\|x\|_p = (\sum_i|x_i|^p)^{1/p}$')

col1, col2 = st.columns(2)

with col1:
  st.header('点群の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  ax.scatter(x[:, 0], x[:, 1], marker = 'o', label = 'Source samples', alpha = a / mx_a, s = marker_size, c = 'r')
  ax.scatter(y[:, 0], y[:, 1], marker = 'x', label = 'Target samples', alpha = b / mx_b, s = marker_size, c = 'b')
  for i in range(n):
    ax.text(x[i, 0] - 0.1, x[i, 1] + 0.1, '$x^{(' + str(i) + ')}$', c = 'r')
  for i in range(m):
    ax.text(y[i, 0] - 0.1, y[i, 1] + 0.1, '$y^{(' + str(i) + ')}$', c = 'b')
  ax.legend()
  ax.set_title('Source and target distributions')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)

with col2:
  st.header('コスト行列の可視化')
  C = distance.cdist(x, y, metric = dist)
  fig, ax = plt.subplots(figsize = figsize)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', '5%', pad = '3%')
  im = ax.imshow(C, cmap = plt.cm.binary)
  fig.colorbar(im, cax = cax)
  ax.set_title('Cost matrix')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)



P_star = ot.emd(a, b, C)
col1, col2 = st.columns(2)

with col1:
  st.header('最適輸送の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  ot.plot.plot2D_samples_mat(x, y, P_star, color = [.5, .5, 1])
  ax.scatter(x[:, 0], x[:, 1], marker = 'o', label = 'Source samples', alpha = a / mx_a, s = marker_size, c = 'r')
  ax.scatter(y[:, 0], y[:, 1], marker = 'x', label = 'Target samples', alpha = b / mx_b, s = marker_size, c = 'b')
  for i in range(n):
    ax.text(x[i, 0] - 0.1, x[i, 1] + 0.1, '$x^{(' + str(i) + ')}$', c = 'r')
  for i in range(m):
    ax.text(y[i, 0] - 0.1, y[i, 1] + 0.1, '$y^{(' + str(i) + ')}$', c = 'b')
  ax.legend()
  ax.set_title('Optimal Transport')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)

with col2:
  st.header('最適輸送行列の可視化')
  fig, ax = plt.subplots(figsize = figsize)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', '5%', pad = '3%')
  im = ax.imshow(P_star, cmap = plt.cm.binary)
  fig.colorbar(im, cax = cax)
  ax.set_title('OT matrix')
  plt.tight_layout()
  buf = BytesIO()
  fig.savefig(buf, format = 'png')
  st.image(buf)
          '''
  st.code(code, language = 'python')
