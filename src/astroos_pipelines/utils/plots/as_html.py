
import numpy as np
import pandas as pd
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_random_samples_as_html(
    dataset,
    label_definitions,
    num_samples_to_display=None,
    seed=None,
    cmap="gist_ncar",
    plot_title="",
    plot_filename=None,
    origin="lower",  # "lower" to mimic imshow(origin="lower")
):
    """
    Plot random samples from a dataset with their labels and features (Plotly native).

    dataset[i] should return:
        (image, label, morph_features, phot_features, metadata)
    """
    import plotly
    print("Using Plotly version:")
    print(plotly.__version__)


    if num_samples_to_display is None:
        num_samples_to_display = min(5, len(dataset))
    else:
        num_samples_to_display = min(num_samples_to_display, len(dataset))

    print(f"Plotting {num_samples_to_display} random samples from dataset of size {len(dataset)}")

    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(dataset), size=num_samples_to_display, replace=False)

    samples = [dataset[i] for i in random_indices]
    random_cutouts = [s[0] for s in samples]
    random_labels = [s[1] for s in samples]
    random_morph_features = [s[2] for s in samples]
    random_phot_features = [s[3] for s in samples]
    random_metadata = [s[4] for s in samples]

    random_samples_info = pd.DataFrame(
        [
            {
                "objectId": str(m.get("OBJECTID", "")),
                "ra": m.get("RA", np.nan),
                "dec": m.get("DEC", np.nan),
                "MIN_RA": m.get("MIN_RA", np.nan),
                "MAX_RA": m.get("MAX_RA", np.nan),
                "MIN_DEC": m.get("MIN_DEC", np.nan),
                "MAX_DEC": m.get("MAX_DEC", np.nan),
            }
            for m in random_metadata
        ]
    )

    bands = ["u", "g", "r", "i", "z"]
    num_rows = num_samples_to_display
    num_cols = len(bands)

    _cmap_map = {
        "gray": "Gray",
        "grey": "Gray",
        "viridis": "Viridis",
        "plasma": "Plasma",
        "magma": "Magma",
        "cividis": "Cividis",
        "inferno": "Inferno",
        "turbo": "Turbo",
    }
    colorscale = _cmap_map.get(str(cmap).lower(), "Viridis")

    total_rows = num_rows * 3 
    base = np.array([3.0, 1.0, 1.0], dtype=float)
    row_heights = np.tile(base / base.sum(), num_rows).tolist()

    specs = []
    for _ in range(num_rows):
        # specs.append([{"colspan": num_cols}] + [None] * (num_cols - 1))  # info row spans
        specs.append([{} for _ in range(num_cols)])  # image row
        specs.append([{} for _ in range(num_cols)])  # morph row
        specs.append([{} for _ in range(num_cols)])  # phot row

    fig = make_subplots(
        rows=total_rows,
        cols=num_cols,
        specs=specs,
        row_heights=row_heights,
        vertical_spacing=0.03,
        horizontal_spacing=0.03,
    )

    if plot_title:
        fig.update_layout(title={"text": plot_title, "x": 0.5})

    def _grey_box(row, col, label="(no data)"):
        # With row/col parameters, Plotly automatically targets the right subplot axes.
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref="y domain",
            x0=0, x1=1, y0=0, y1=1,
            line={"width": 0},
            fillcolor="lightgray",
            layer="below",
            row=row, col=col,
        )
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="x domain",
            yref="y domain",
            text=label,
            showarrow=False,
            font={"size": 12, "color": "dimgray"},
            row=row, col=col,
        )

    for i in range(num_rows):
        base_row = i * 3 
        # info_row = base_row + 1
        img_row = base_row + 1 
        morph_row = base_row + 2  
        phot_row = base_row + 3

        label_id = int(random_labels[i]) if random_labels[i] is not None else -1
        label_classname = (
            label_definitions.iloc[label_id]["long_name"]
            if (label_definitions is not None and 0 <= label_id < len(label_definitions))
            else ""
        )
        objid = str(random_samples_info.iloc[i]["objectId"]) if not random_samples_info.empty else ""
        label_full = f"Label [{label_classname}] objectId [{objid}]"

        # info row annotation spanning all columns
        # fig.add_annotation(
            # x=0.5, y=0.5,
            # xref="x domain",
            # yref="y domain",
            # text=label_full,
            # showarrow=False,
            # font={"size": 18},
            # row=info_row, col=1,
        # )
        # for c in range(1, num_cols + 1):
            # fig.update_xaxes(visible=False, row=info_row, col=c)
            # fig.update_yaxes(visible=False, row=info_row, col=c)

        for j in range(num_cols):
            print(
                f"Plotting sample {i}, band {bands[j]}, objectId: {random_samples_info.iloc[i]['objectId']}, "
                f"ra: {random_samples_info.iloc[i]['ra']}, dec: {random_samples_info.iloc[i]['dec']}"
            )

            # ---- image (raw pixels, batlow, RA/Dec extent, y reversed) ----
            if random_cutouts[i] is not None:
                cutout = np.asarray(random_cutouts[i][j], dtype=np.float32)


                # extent should be (min_ra, max_ra, min_dec, max_dec)
                extent = (
                        float(random_metadata[i]["MIN_RA"]),
                        float(random_metadata[i]["MAX_RA"]),
                        float(random_metadata[i]["MIN_DEC"]),
                        float(random_metadata[i]["MAX_DEC"]),
                        )

                min_ra, max_ra, min_dec, max_dec = extent
                ny, nx = cutout.shape[-2], cutout.shape[-1]

                # pixel-center coordinates in world units
                x = np.linspace(min_ra, max_ra, nx)
                y = np.linspace(min_dec, max_dec, ny)

                hm = go.Heatmap(
                        z=cutout,
                        x=x,
                        y=y,
                        colorscale=colorscale,
                        zsmooth=False,         # IMPORTANT: no smoothing
                        showscale=False,
                        hovertemplate="RA=%{x:.6f}°<br>Dec=%{y:.6f}°<br>val=%{z:.6g}<extra></extra>",
                        )
                fig.add_trace(hm, row=img_row, col=j + 1)

                fig.update_xaxes(
                        # title_text="RA °", 
                        row=img_row, 
                        col=j + 1, 
                        nticks=1,
                        # autorange="reversed" 
                        )
                fig.update_xaxes(
                        title_text="RA °", 
                        row=img_row, 
                        col=1, 
                        nticks=3,
                        # autorange="reversed" 
                        )
                fig.update_yaxes(
                        # title_text="Dec °",
                        row=img_row,
                        col=j + 1,
                        nticks=1,
                        # autorange="reversed",
                        )
                fig.update_yaxes(
                        title_text="Dec °",
                        row=img_row,
                        col=1,
                        nticks=3,
                        # autorange="reversed",
                        )

                # subplot title per band
                fig.add_annotation(
                        x=0.5,
                        y=1.05,
                        xref="x domain",
                        yref="y domain",
                        text=f"band: {bands[j]}",
                        showarrow=False,
                        font={"size": 14},
                        row=1,
                        col=j + 1,
                        )
            else:
                _grey_box(img_row, j + 1, "(no data)")
                fig.update_xaxes(visible=False, row=img_row, col=j + 1)
                fig.update_yaxes(visible=False, row=img_row, col=j + 1)

            # -------- morph --------
            if random_morph_features[i] is not None:
                features_morph = np.array(random_morph_features[i][j], dtype=np.float32)
                fig.add_trace(
                    go.Bar(x=["C", "A", "S", "H"], y=features_morph, marker_color="skyblue", showlegend=False),
                    row=morph_row, col=j + 1,
                )
                fig.update_yaxes(range=[0.0, 1.0], row=morph_row, col=1, title_text="Norm")
                fig.update_xaxes(row=morph_row, col=1, title_text="Morphometry")
            else:
                _grey_box(morph_row, j + 1, "(no data)")
                fig.update_xaxes(visible=False, row=morph_row, col=j + 1)
                fig.update_yaxes(visible=False, row=morph_row, col=j + 1)

            # -------- phot --------
            if random_phot_features[i] is not None:
                features_phot = np.array(random_phot_features[i][j], dtype=np.float32)
                lower = 0.0 if np.nanmin(features_phot) > 0 else float(np.nanmin(features_phot) * 2)
                upper = float(np.nanmax(features_phot) * 2) if np.nanmax(features_phot) > 0 else 0.0
                fig.add_trace(
                    go.Bar(x=["x1", "x2", "x3", "x4"], y=features_phot, marker_color="skyblue", showlegend=False),
                    row=phot_row, col=j + 1,
                )
                fig.update_yaxes(range=[lower, upper], row=phot_row, col=1, title_text="Value")
                fig.update_xaxes(row=phot_row, col=1, title_text="Photometry")
            else:
                _grey_box(phot_row, j + 1, "(no data)")
                fig.update_xaxes(visible=False, row=phot_row, col=j + 1)
                fig.update_yaxes(visible=False, row=phot_row, col=j + 1)

    fig.update_layout(
        height=int(250 * num_rows * 4),
        width=1400,
        margin=dict(l=20, r=20, t=60, b=20),
    )


    output_dir = os.path.join(dataset.get_dataset_dir(), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    if plot_filename is None:
        plot_filename = f"{dataset.get_dataset_dir()}/plots/random_samples_{num_samples_to_display}.html"
    else:
        print(f"Using provided plot filename: {plot_filename}")

    fig.write_html(plot_filename, include_plotlyjs="cdn")
    print(f"Plot saved to {plot_filename}")

    return plot_filename, num_samples_to_display



