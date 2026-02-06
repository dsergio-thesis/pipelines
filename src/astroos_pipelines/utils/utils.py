
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

import time

# device = torch.device("cpu") #default device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker


# Batlow
import cmcrameri.cm as cmc

def train_image(model, dataloader, opt, n_epochs=1, loss_fn=F.cross_entropy, device=device):
    
    model.train()
    
    start = time.time()
    
    for i in range(n_epochs):
        error = 0
        loss_ = 0.0
        sample_count = 0
        
        for j, batch in enumerate(dataloader):
            if j % 10 == 0:
                # print(f"mini batch: {j} | elapsed time: {time.time()-start:.2f} seconds")
                pass
            
            # print(batch.shape)
            
            X, y, _, _ = batch

            # print("X")
            # print(X.shape)

            # print(X)
            
            X = X.to(device) #Torch dataloader returns images into NCHW format

            # print(f"X shape: {X.shape}")
            # print(f"y shape: {y.shape}")

            y = y.to(device)
            
            Z = model(X)
#            print(f"Z shape: {Z.size()}")
#            print(f"y shape: {y.size()}")
            loss = loss_fn(Z, y)
            loss.backward()  
            opt.step()

            
            opt.zero_grad()
            
            with torch.no_grad():
                loss_ += loss * X.shape[0]
                error += (Z.argmax(dim=1) != y).sum().to("cpu")
                
                sample_count += X.shape[0]

    return loss_ / sample_count, error/sample_count


def test_image(model, dataloader, loss_fn=F.cross_entropy, device=device):

    model.eval()

    with torch.no_grad():
        error = torch.tensor(0, device=device)
        loss = torch.tensor(0.0, device=device)
        sample_count = 0
        for i, batch in enumerate(dataloader):
            X, y, _, _ = batch
            X = X.to(device) #Torch dataloader returns images into NCHW format
            y = y.to(device)

            Z = model(X)
            loss += loss_fn(Z, y) * X.shape[0]
            error += (Z.argmax(dim=1) != y).sum().to("cpu")
            sample_count += X.shape[0]
        return loss/sample_count, error/sample_count

def predict(model, dataloader, device=device):

    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X, _ = batch
            X = X.to(device) #Torch dataloader returns images into NCHW format

            Z = model(X)
            predictions.append(Z.argmax(dim=1).to("cpu"))
    predictions = torch.cat(predictions, dim=0)
    return predictions

    
import matplotlib.pyplot as plt
import numpy as np

def plt_loss_error(train_l=None, test_l=None, train_e=None, test_e=None):
    # Convert tensors to numpy
    train_loss = [t.cpu().numpy() for t in train_l]
    test_loss = [t.cpu().numpy() for t in test_l]
    train_error = [t.cpu().numpy() for t in train_e]
    test_error = [t.cpu().numpy() for t in test_e]
    
    x = np.arange(len(train_l))
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    # Plot loss on left y-axis
    ax1.plot(x, train_loss, label="Train Loss", color="tab:blue", linestyle='-')
    ax1.plot(x, test_loss, label="Test Loss", color="tab:blue", linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    # Create second y-axis for error
    ax2 = ax1.twinx()
    ax2.plot(x, train_error, label="Train Error", color="tab:red", linestyle='-')
    ax2.plot(x, test_error, label="Test Error", color="tab:red", linestyle='--')
    ax2.set_ylabel("Error", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    
    plt.title("Training Loss and Error")
    plt.tight_layout()
    plt.savefig("plot_error_loss.png", dpi=300)
    plt.close()


def plot_survey_images(data, nrows=2, ncols=5, figsize=[8,4], title='', idx=0, each_label=None):
    '''Plot an array of images'''

    c = 0

    if each_label is None:
        each_label = [f'{i+1}' for i in range(nrows)]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # fig.subplots_adjust(left=0.15, right=0.95, wspace=0.01, hspace=0.01, bottom=0.15)

    for i in range(nrows):
        for j in range(ncols):

            if (idx + i >= data.shape[0]):
                break            
                
            if j == 0:
                ax[i][j].set_ylabel(each_label[i], fontsize=12, rotation=0, labelpad=30)
            # if j == 0:
            #     ax[i, j].text(-0.1, 0.5, each_label[i], va='center', ha='left', transform=ax[i, j].transAxes)


            ax[i][j].imshow(data[idx + i][j], cmap='gist_ncar')

        if (i == 0):
            ax[i][j].xaxis.set_major_formatter(plt.NullFormatter())
        if (j != 0):
            ax[i][j].yaxis.set_major_formatter(plt.NullFormatter())

        # ax[i][j].imshow(data[idx + i][j], cmap='gist_ncar')
            
        # if (j < ncols):
        #     j += 1
        # if (j == ncols):
        #     j = 0
        #     if (i < nrows - 1):
        #         i += 1
        c += 1

    fig.suptitle(f"{title} idx={idx}", fontsize=16)

    ax[nrows-1][int(ncols/2)].set_xlabel('$band$')   


import pandas as pd

def label_definitions(classification_labels_file):

    label_definitions = pd.read_csv(classification_labels_file)

    # add column in the beginning, which is an index
    if ("label_index" not in label_definitions.columns):
        label_definitions.insert(0, 'index', range(1, len(label_definitions) + 1))
    label_definitions.to_csv(classification_labels_file, index=False)

    # add index for fast lookup
    label_definitions.set_index('morph_type', inplace=True)

    return label_definitions


import warnings
warnings.filterwarnings("error", message="Mean of empty slice")


def plot_random_samples_from_dataset(
        dataset, 
        label_definitions,
        num_samples_to_display=5, 
        seed=None,
        cmap='gist_ncar',
        plot_title="",
        plot_filename=None,
        simple_plot=False,
        ):
    
    print(f"Plotting {num_samples_to_display} random samples from dataset of size {len(dataset)}")
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    random_indices = np.random.choice(len(dataset), size=num_samples_to_display, replace=False)

    print("First index:")
    print(dataset[0])

    # print("Random indices:", random_indices)
    random_samples = [dataset[i][0] for i in random_indices]
    random_labels = [dataset[i][1] for i in random_indices]
    random_morph_features = [dataset[i][2] for i in random_indices]
    random_image_bounds = [(dataset[i][4]['MIN_RA'], dataset[i][4]['MAX_RA'],
                            dataset[i][4]['MIN_DEC'], dataset[i][4]['MAX_DEC'])
                           for i in random_indices]
    random_main_ids = [dataset[i][4]['MAIN_ID'] for i in random_indices]
    random_ras = [dataset[i][4]['RA'] for i in random_indices]
    random_decs = [dataset[i][4]['DEC'] for i in random_indices]
    random_rvz = [dataset[i][4]['rvz_redshift'] for i in random_indices]
    for i in range(len(random_samples)):
        print(random_samples[i].shape)
        print("mean: ", random_samples[i].mean())

    # labels = label_definitions("./sdss_morph_types_info.csv")
    # print(labels)
    # print("Random labels:", random_labels)
    # print("Random bounds:", random_image_bounds)
    # print("random samples: ", np.array(random_samples).shape)
    # print("label definitions:", label_definitions)


    # if sample_information_file is not None:
    #     random_samples_info = pd.read_csv(sample_information_file)[['main_id', 'rvz_redshift', 'galdim_majaxis', 'galdim_minaxis', 'galdim_angle']]
    #     # get only the rows corresponding to random_indices
    #     random_samples_info = random_samples_info.iloc[random_indices]

    print("random_image_bounds:", random_image_bounds)

    random_samples_info = pd.DataFrame()

    for i in range(len(random_main_ids)):
        row = pd.DataFrame([{
            'main_id': random_main_ids[i],
            'rvz_redshift': random_rvz[i],
            'ra': random_ras[i],
            'dec': random_decs[i],
        }])
        random_samples_info = pd.concat([random_samples_info, row], ignore_index=True)


    for i in random_samples_info.index:
        if random_samples_info.at[i, 'rvz_redshift'] == '--':
            random_samples_info.at[i, 'rvz_redshift'] = 0.0 # do something else here

    summary = []
    for i in range(num_samples_to_display):
        summary.append(
            (random_labels[i], 
             label_definitions.iloc[int(random_labels[i])]["long_name"],
             random_samples_info.iloc[i] if not random_samples_info.empty else "",
             random_morph_features[i] if random_morph_features else ""
            ))
        
    # print("Random samples summary (label index, sample info):")
    # for item in summary:
    #     print(item)

    # print(random_morph_features)
    # print(random_samples_info)
    # print(label_definitions)
    # print(summary)


    bands = ['u', 'g', 'r', 'i', 'z']
    # bands = ['u', 'g']
    num_rows = num_samples_to_display
    num_cols = len(bands) 

    fig = plt.figure(figsize=(16, num_samples_to_display * 4), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=24)
    gs = gridspec.GridSpec(
        num_rows * 3, 
        num_cols, 
        figure=fig,
        width_ratios=[1.0] * num_cols,
        height_ratios=([.25] + [3.0] + [1.0]) * num_rows,
    )

    for i in range(num_rows):  # rows
        # i += 1
        plot_index = i * 3 - 3

        label_classname = label_definitions.iloc[int(random_labels[i-1])]["long_name"]
        info = random_samples_info.iloc[i-1]['main_id'] if not random_samples_info.empty else ""

        redshift = random_samples_info.iloc[i-1]['rvz_redshift']
        ax_info = fig.add_subplot(gs[plot_index, :])

        if simple_plot:
            label_full = f"{info}"
        else:
            label_full = f"{label_classname} {info} (z={redshift})"

        ax_info.text(0.5, 0.0, label_full,
                    rotation=0, ha='center', va='center', fontsize=18)
        ax_info.axis('off')

        

        for j in range(num_cols):       # image columns
            
            sample = random_samples[i-1][j]
            
            extent = random_image_bounds[i-1] if random_image_bounds is not None else None
            print(f"Plotting sample {i-1}, band {bands[j]}, main_id: {random_samples_info.iloc[i-1]['main_id']}, ra: {random_samples_info.iloc[i-1]['ra']}, dec: {random_samples_info.iloc[i-1]['dec']}")
            # print(f"extent.shape: {extent.shape} extent: {extent} sample.shape: {sample.shape}" \
            #       f" ra_diff: {extent[1]-extent[0]} dec_diff: {extent[3]-extent[2]}")
            
            # ra_min, ra_max, dec_min, dec_max = extent
            # ra_center = 0.5*(ra_min + ra_max)
            # ra_diff = dec_max - dec_min  # make RA span = Dec span for plotting
            # extent_plot = [ra_center - ra_diff/2, ra_center + ra_diff/2, dec_min, dec_max]

            ax = fig.add_subplot(gs[plot_index + 1, j])
            ax.imshow(sample,
                    cmap=cmap,
                    extent=extent if extent is not None else None,
                    aspect='auto'

                    # cmap='plasma',
                    # cmap=cmc.batlow, # better for colorblind
            )
            # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            ax.set_title(f"band: {bands[j]}", fontsize=14)
            # ax.axis('off')
            ax.set_xlabel("RA °", fontdict={'fontsize': 18}, labelpad=14)
            ax.set_ylabel("Dec °", fontdict={'fontsize': 18}, labelpad=14)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))

            ax.tick_params(axis='x', labelsize=12, pad=2)
            ax.tick_params(axis='y', labelsize=12, pad=2)

            
            # ax_label.text(0.5, 0.5, f"features: {1}",
            #         rotation=0, ha='center', va='center', fontsize=12)
            ax_cash = fig.add_subplot(gs[plot_index + 2, j])
            features = random_morph_features[i-1][j]
            # features = np.concatenate([features, np.array([random_samples_info.iloc[i-1]['rvz_redshift']])], axis=0)  # for R
            features = np.array(features, dtype=np.float32)
            # print(f"features: {features} features.shape: {features.shape}")
            x = np.array([0, 1, 2, 3])

            # set features to random numbers
            features = np.random.uniform(0, 1, size=(4)) 

            ax_cash.bar(x, features, width=0.25, color='skyblue', edgecolor='black')
            # ax_cash.scatter(x, features, marker='o', color='blue')
            ax_cash.set_xticks(x)
            ax_cash.set_yticks(np.linspace(0.0, 1.0, num=3))
            ax_cash.set_xticklabels(['C', 'A', 'S', 'H'])
            ax_cash.tick_params(axis='x', labelsize=14, pad=2)
            ax_cash.set_ylim(0.0, 1.0)
            ax_cash.set_xlabel('Morphometric feature', fontsize=12)
            ax_cash.set_ylabel('Norm', fontsize=12)


            # ax_cash.set_aspect(2.0)
            # ax_cash.set_adjustable('datalim')
            # ax_cash.axis('off')
            
    # fig.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.0)
    # Add horizontal lines between rows
    # for i in range(1, num_rows * 3):
    #     # compute y in figure coordinates
    #     if (i % 3) == 0:
    #         y = 1 - i/(num_rows * 3)  # adjust if needed
    #         line = mlines.Line2D([0, 1], [y, y], transform=fig.transFigure, color='black', linewidth=1)
    #         fig.lines.append(line)
    if plot_filename:
        plt.savefig(plot_filename, dpi=300)
    else:
        plt.show()
