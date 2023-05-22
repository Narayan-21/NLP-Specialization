import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError('x and y must be same size.')
    cov = np.cov(x,y) # Returns the covariance matrix between x and y.
    pearson = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
    radius_x = np.sqrt(1+pearson)
    radius_y = np.sqrt(1-pearson)
    ellipse = Ellipse((0,0), width=radius_x*2, height=radius_y*2, facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0,0]) * n_std # Standard deviation of x
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1,1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


data = pd.read_csv(
    'Sentiment Analysis with Naive Bayes/Naive Bayes visualization/bayes_features.csv')

fig, ax = plt.subplots(figsize=(8,8))
colors = ['red', 'green']
sentiments = ['negative', 'positive']
index = data.index

for sentiment in data.sentiment.unique():
    ix = index[data.sentiment == sentiment]
    ax.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*', label=sentiments[int(sentiment)])

ax.legend(loc='best')

plt.xlim(-250, 0)
plt.ylim(-250, 0)

plt.xlabel('Positive')
plt.ylabel('Negative')
plt.show()


# Using confidence interval to interpret Naive Bayes:
# Plot the samples using columns 1 and 2 of the matrix
fig2, ax2 = plt.subplots(figsize=(8, 8))

# Color base on sentiment
for sentiment in data.sentiment.unique():
    ix = index[data.sentiment == sentiment]
    ax2.scatter(data.iloc[ix].positive, data.iloc[ix].negative, c=colors[int(sentiment)], s=0.1, marker='*', label=sentiments[int(sentiment)])

# Custom limits for this chart
plt.xlim(-200, 40)
plt.ylim(-200, 40)

plt.xlabel("Positive")  # x-axis label
plt.ylabel("Negative")  # y-axis label

data_pos = data[data.sentiment == 1]  # Filter only the positive samples
data_neg = data[data.sentiment == 0]  # Filter only the negative samples

# Print confidence ellipses of 2 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax2, n_std=2, edgecolor='black', label=r'$2\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax2, n_std=2, edgecolor='orange')

# Print confidence ellipses of 3 std
confidence_ellipse(data_pos.positive, data_pos.negative, ax2, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax2, n_std=3, edgecolor='orange', linestyle=':')
ax2.legend(loc='lower right')

plt.show()
