import numpy as np
import pandas as pd
from scipy import stats
import copy
import re
import altair as alt
from timeshap.plot.utils import multi_plot_wrapper

from typing import Union, List, Optional, Tuple


def plot_event_heatmap(event_data: pd.DataFrame, plot_size: Tuple[int, int]):
    """Plots local event explanations

    Parameters
    ----------
    event_data: pd.DataFrame
        Event global explanations
    """
    event_data = copy.deepcopy(event_data)
    # extract digit to order df by - this is redundant but gives security to the method
    event_data["idx"] = event_data["Feature"].apply(
        lambda x: event_data.shape[0]
        if "Pruned Events" in x
        else int(re.findall(r"\d+", x)[0]) - 1
    )
    event_data = event_data.sort_values("idx")[["Shapley Value", "Feature"]]

    c_range = [
        "#5f8fd6",
        "#99c3fb",
        "#f5f5f5",
        "#ffaa92",
        "#d16f5b",
    ]

    event_data["row"] = event_data["Feature"].apply(
        lambda x: event_data.shape[0]
        if x == "Pruned Events"
        else -eval(x.split(":")[0][6:])
    )
    event_data["column"] = event_data["Feature"].apply(lambda x: 1)
    event_data["rounded"] = event_data["Shapley Value"].apply(lambda x: round(x, 3))
    event_data["rounded_str"] = event_data["Shapley Value"].apply(
        lambda x: "0.000" if round(x, 3) == 0 else str(round(x, 3))
    )
    event_data["rounded_str"] = event_data["rounded_str"].apply(
        lambda x: f"{x}0" if len(x) == 4 else x
    )

    c = alt.Chart().encode(
        y=alt.Y(
            "Feature:O",
            axis=alt.Axis(
                domain=False,
                labelFontSize=15,
                title="Event",
                titleFontSize=15,
                titleX=-49,
            ),
            sort=list(event_data["Feature"].values),
        ),
    )

    a = c.mark_rect().encode(
        x=alt.X("column:O", axis=alt.Axis(title="Shapley Value", titleFontSize=15)),
        color=alt.Color(
            "rounded",
            title=None,
            legend=alt.Legend(
                gradientLength=225,
                gradientThickness=10,
                orient="right",
                labelFontSize=15,
            ),
            # scale=alt.Scale(domain=[-0.5, 0.5], range=c_range),
        ),
    )
    b = c.mark_text(
        align="right", baseline="middle", dx=18, fontSize=15, color="#798184"
    ).encode(
        x=alt.X(
            "column:O",
            axis=alt.Axis(labels=False, title="Shapley Value", domain=False, titleX=43),
        ),
        text="rounded_str",
    )

    event_plot = alt.layer(a, b, data=event_data).properties(
        width=plot_size[0], height=plot_size[1]
    )
    return event_plot


def plot_feat_barplot(
    feat_data: pd.DataFrame,
    top_x_feats: int = 15,
    plot_features: Optional[dict] = None,
    plot_size: Tuple[int, int] = (190, 225),
):
    """Plots local feature explanations

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature explanations

    top_x_feats: int
        The number of feature to display.

    plot_features: dict
        Dict containing mapping between model features and display features
    """
    feat_data = copy.deepcopy(feat_data)
    if plot_features:
        plot_features["Pruned Events"] = "Pruned Events"
        feat_data["Feature"] = feat_data["Feature"].apply(lambda x: plot_features[x])

    feat_data["sort_col"] = feat_data["Shapley Value"].apply(lambda x: abs(x))

    if top_x_feats is not None and feat_data.shape[0] > top_x_feats:
        sorted_df = feat_data.sort_values("sort_col", ascending=False)
        cutoff_contribution = abs(sorted_df.iloc[4]["Shapley Value"])
        feat_data = feat_data[
            np.logical_or(
                feat_data["Explanation"] >= cutoff_contribution,
                feat_data["Explanation"] <= -cutoff_contribution,
            )
        ]

    a = (
        alt.Chart(feat_data)
        .mark_bar(size=15, thickness=1)
        .encode(
            y=alt.Y(
                "Feature",
                axis=alt.Axis(
                    title="Feature", labelFontSize=15, titleFontSize=15, titleX=-61
                ),
                sort=alt.SortField(field="sort_col", order="descending"),
            ),
            x=alt.X(
                "Shapley Value",
                axis=alt.Axis(
                    grid=True, title="Shapley Value", labelFontSize=15, titleFontSize=15
                ),
            ),
            # scale=alt.Scale(domain=[-0.5, 0.5])),
        )
    )

    line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#798184").encode(x="x")

    feature_plot = (a + line).properties(width=plot_size[0], height=plot_size[1])
    return feature_plot


def plot_cell_level(
    cell_data: pd.DataFrame,
    model_features: List[str],
    plot_features: dict,
    plot_parameters: Optional[dict] = None,
):
    """Plots local feature explanations

    Parameters
    ----------
    cell_data: pd.DataFrame
        Cell level explanations

    model_features: int
        The number of feature to display.

    plot_features: dict
        Dict containing mapping between model features and display features

    plot_parameters: dict
        Dict containing optinal plot parameters
            'height': height of the plot, default 225
            'width': width of the plot, default 200
            'axis_lims': plot Y domain, default [-0.5, 0.5]
            'FontSize': plot font size, default 15
    """
    c_range = [
        "#5f8fd6",
        "#99c3fb",
        "#f5f5f5",
        "#ffaa92",
        "#d16f5b",
    ]
    unique_events = [
        x
        for x in np.unique(cell_data["Event"].values)
        if x not in ["Other Events", "Pruned Events"]
    ]
    sort_events = sorted(
        unique_events, key=lambda x: re.findall(r"\d+", x)[0], reverse=True
    )
    unique_feats = [
        x
        for x in np.unique(cell_data["Feature"].values)
        if x not in ["Other Features", "Pruned Events"]
    ]
    if plot_features:
        plot_features = copy.deepcopy(plot_features)
        sort_features = [plot_features[x] for x in model_features if x in unique_feats]
        if "Other Features" in np.unique(cell_data["Feature"].values):
            plot_features["Other Features"] = "Other Features"

        if "Pruned Events" in np.unique(cell_data["Feature"].values):
            plot_features["Pruned Events"] = "Pruned Events"

        cell_data["Feature"] = cell_data["Feature"].apply(lambda x: plot_features[x])
    else:
        sort_features = [x for x in model_features if x in unique_feats]

    cell_data["rounded"] = cell_data["Shapley Value"].apply(lambda x: round(x, 3))
    cell_data["rounded_str"] = cell_data["Shapley Value"].apply(
        lambda x: "0.000" if round(x, 3) == 0 else str(round(x, 3))
    )
    cell_data["rounded_str"] = cell_data["rounded_str"].apply(
        lambda x: f"{x}0" if len(x) == 4 else x
    )

    filtered_cell_data = cell_data[
        ~np.logical_and(
            cell_data["Event"] == "Pruned Events",
            cell_data["Feature"] == "Pruned Events",
        )
    ]

    if plot_parameters is None:
        plot_parameters = {}
    height = plot_parameters.get("height", 225)
    width = plot_parameters.get("width", 200)
    axis_lims = plot_parameters.get("axis_lim", [-0.5, 0.5])
    fontsize = plot_parameters.get("FontSize", 15)

    c = alt.Chart().encode(
        y=alt.Y(
            "Feature",
            axis=alt.Axis(domain=False, labelFontSize=fontsize, title=None),
            sort=sort_features,
        ),
    )

    a = c.mark_rect().encode(
        x=alt.X("Event", axis=alt.Axis(titleFontSize=15), sort=sort_events),
        color=alt.Color(
            "rounded",
            title=None,
            legend=alt.Legend(
                gradientLength=height,
                gradientThickness=10,
                orient="right",
                labelFontSize=fontsize,
            ),
            scale=alt.Scale(domain=axis_lims, range=c_range),
        ),
    )
    b = c.mark_text(
        align="right", baseline="middle", dx=18, fontSize=15, color="#798184"
    ).encode(
        x=alt.X(
            "Event",
            sort=sort_events,
            axis=alt.Axis(
                orient="top",
                title="Shapley Value",
                domain=False,
                titleY=height + 20,
                titleX=172,
                labelAngle=30,
                labelFontSize=fontsize,
            ),
        ),
        text="rounded_str",
    )

    cell_plot = alt.layer(a, b, data=filtered_cell_data).properties(
        width=math.ceil(0.8 * width), height=height
    )

    if "Pruned Events" in np.unique(cell_data["Event"].values):
        # isolate the pruned contribution
        df_prun = cell_data[
            np.logical_and(
                cell_data["Event"] == "Pruned Events",
                cell_data["Feature"] == "Pruned Events",
            )
        ]
        assert df_prun.shape == (1, 5)
        prun_rounded_str = df_prun.iloc[0]["rounded_str"]
        prun_rounded = df_prun.iloc[0]["rounded"]
        df_prun = pd.DataFrame(
            [
                [
                    ["Pruned", "Events"],
                    "Other features",
                    prun_rounded,
                    prun_rounded_str,
                ],
            ],
            columns=["Event", "Feature", "rounded", "rounded_str"],
        )

        c = alt.Chart().encode(
            y=alt.Y("Feature", axis=alt.Axis(labels=False, domain=False, title=None)),
        )

        a = c.mark_rect().encode(
            x=alt.X("Event", axis=alt.Axis(titleFontSize=fontsize)),
            color=alt.Color(
                "rounded",
                title=None,
                legend=None,
                scale=alt.Scale(domain=axis_lims, range=c_range),
            ),
        )
        b = c.mark_text(
            align="right", dx=18, baseline="middle", fontSize=fontsize, color="#798184"
        ).encode(
            x=alt.X(
                "Event",
                axis=alt.Axis(
                    labelOffset=24,
                    labelPadding=30,
                    orient="top",
                    title=None,
                    domain=False,
                    labelAngle=0,
                    labelFontSize=fontsize,
                ),
            ),
            text="rounded_str",
        )

        cell_plot_prun = alt.layer(a, b, data=df_prun).properties(
            width=width / 3, height=height
        )

        cell_plot = alt.hconcat(cell_plot_prun, cell_plot).resolve_scale(
            color="independent"
        )
    return cell_plot
