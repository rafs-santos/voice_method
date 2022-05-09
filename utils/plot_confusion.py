import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.colors import ListedColormap


def plot_confusion_aux(conf_mtx):
  # Column and Line
  c_ = conf_mtx.shape[1]
  l_ = conf_mtx.shape[0]

  conf_plot = np.zeros((l_ + 1, c_ + 1))
  conf_plot[:l_, :c_] = conf_mtx
  precision = 0
  recall = 0
  accuracy = 0
  total_data = 0
  # Calculate precision recal
  for i in range(l_):
    for j in range(c_):
      if i==j:
        accuracy = accuracy + conf_mtx[i,j]
        total_data = total_data + sum(conf_mtx[:,j])

        # Precision
        conf_plot[i, c_] = conf_mtx[i,j]/sum(conf_mtx[i,:])
        # Recall
        conf_plot[l_, j] = conf_mtx[i,j]/sum(conf_mtx[:,j])

  conf_plot[l_, c_] = accuracy/total_data
  return conf_plot, total_data

def plot_confusion(conf_mtx, ax, target_names = None):
  # Column and Line
  c_ = conf_mtx.shape[1]
  l_ = conf_mtx.shape[0]
  ################### Make color plot and style of the figure
  # Make the color map:
  my_cmp = ListedColormap(['mediumspringgreen', 'darksalmon', 'azure', 'deepskyblue'])
  color_plot = np.zeros((l_+1, c_+1))
  for i in range(c_+1):
    for j in range(l_+1):
      if i < l_ and j < l_:
        if i == j:
          color_plot[i,j] = 0
        else:
          color_plot[i,j] = 1
      else:
        color_plot[i,j] = 2

  # Accuracy color = 3
  color_plot[-1,-1] = 3
  # Plot first image
  im = ax.imshow(color_plot, interpolation='nearest', cmap=my_cmp, zorder=0)

  if target_names == None:
    # Put 0, 1, 2 in names
    target_names = []
    for i in range(c_):
      aux_name = str(i)
      target_names.append(fr"{aux_name}")
  else:
      target_names.append(f" ")

  # Set Tick for axis with target_names
  ax.set(xticks=np.arange(color_plot.shape[1]),
            yticks=np.arange(color_plot.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=target_names, yticklabels=target_names)
  # Style of tick label (bold)
  tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
  [label.set_fontweight('bold') for label in tick_labels]
  # Font and style of axis label
  fontdict = {'weight': 'bold',
      'size': 16,
  }
  ax.set_xlabel(r"True label", fontdict=fontdict)
  ax.set_ylabel(r"Predicted label", fontdict=fontdict)

  # Minor ticks
  ax.set_xticks(np.arange(-.5, c_, 1), minor=True)
  ax.set_yticks(np.arange(-.5, l_, 1), minor=True)
  ax.grid(which='minor', color='k', linestyle='-', linewidth=1.1)
  ax.tick_params(axis=u'both', which=u'both',length=0)

  # Put larger linewidth separate confusion matrix from metrics
  x_tick_pos = ax.xaxis.get_minorticklocs()
  y_tick_pos = ax.yaxis.get_minorticklocs()

  ax.axvline(x=x_tick_pos[c_], color='k', linestyle='-', linewidth=3.5)
  ax.axhline(y=y_tick_pos[l_], color='k', linestyle='-', linewidth=3.5)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

  ################### Put the values on the matrix
  values_mtx, total_data = plot_confusion_aux(conf_mtx)
  # Loop over data dimensions and create text annotations.
  # pad value for vertical position
  skip_v = -.1
  for i in range(values_mtx.shape[0]):
    for j in range(values_mtx.shape[1]):
      if i < l_ and j < c_:
          string_format0 = r"$\mathbf{{{:.0f}}}$"
          string_format1 = "\n\n"+r"${:.1f}\%$"
          ax.text(j, i + skip_v, string_format0.format(values_mtx[i, j]),
                  ha="center", va="center", weight='bold',
                  color="black")
          ax.text(j, i + skip_v, string_format1.format((values_mtx[i, j]/total_data)*100),
              ha="center", va="center", fontsize='small',
              color="black")
      else:
        string_format0 = r"$\mathbf{{{:.1f}\%}}$"
        string_format1 = "\n\n"+r"${:.1f}\%$"
        ax.text(j, i + skip_v, string_format0.format(values_mtx[i, j]*100),
                ha="center", va="center", weight='bold',
                color="green")
        ax.text(j, i + skip_v, string_format1.format(100-values_mtx[i,j]*100),
                ha="center", va="center", fontsize='small',
                color="red")


