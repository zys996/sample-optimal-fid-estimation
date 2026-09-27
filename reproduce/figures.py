"""Paper plotting styles and layouts; all plotted statistics are recomputed."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter, LogLocator, LogFormatterMathtext, NullFormatter, MaxNLocator
import numpy as np
import pandas as pd

METHODS = ["empirical_plugin", "fid_infinity_ols_order1", "fid_infinity_ols_order2",
           "fid_infinity_ols_order3", "general_adaptive"]

LABELS = dict(zip(METHODS, ["Empirical", r"OLS$_1$", r"OLS$_2$", r"OLS$_3$", "RTD"]))

COLORS = dict(zip(METHODS, ["#55616C", "#0072B2", "#D55E00", "#8A56A2", "#009E73"]))

MARKERS = dict(zip(METHODS, ["s", "^", "v", "D", "o"]))

DIMS = [200, 300, 400, 600, 900, 1200, 1600, 2048, 2500, 3000]

BUDGETS = [50000, 60000, 80000, 100000]

RANDOM = ["haar_log_uniform", "spiked_bulk", "rotated_toeplitz"]

RANK = ["rank_fraction_025", "rank_fraction_050", "rank_fraction_075"]

CONDITION = ["ill_conditioned_kappa_1e2", "ill_conditioned_kappa_1e4", "ill_conditioned_kappa_1e6"]

REFERENCES = [
    {"power": 2.0, "style": (0, (5, 3)), "label": r"$d^2$ guide"},
    {"power": 3.0, "style": (0, (1.5, 2)), "label": r"$d^3$ guide"},
    {"power": 0.5, "style": (0, (5, 2, 1, 2)), "label": r"$d^{1/2}$ guide"},
]

def styles():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10.5,
        "axes.linewidth": 0.7, "axes.edgecolor": "#9AA2A9",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "grid.color": "#DCE1E5", "grid.linewidth": 0.5,
    })

def plot_grid(data, row_key, row_values, row_labels, output, reference=False, yscale="log"):
    if yscale not in {"log", "linear"}:
        raise ValueError("yscale must be log or linear")
    if yscale == "log" and data.lower_mean_minus_sd.le(0).any():
        raise ValueError("Mean-minus-SD is nonpositive; use a linear y axis to show the full band")
    styles()
    nrows = len(row_values)
    height = 3.5 if nrows == 1 else (7.5 if reference else 7.15)
    fig, axes = plt.subplots(nrows, 4, figsize=(9.7, height), sharex=True, sharey="row", squeeze=False)
    for row, (value, row_label) in enumerate(zip(row_values, row_labels)):
        for col, budget in enumerate(BUDGETS):
            ax = axes[row, col]
            panel = data[data[row_key].eq(value) & data.N.eq(budget)]
            for method in METHODS:
                line = panel[panel.method.eq(method)].sort_values("dimension")
                assert line.dimension.tolist() == DIMS
                x = line.dimension.to_numpy()
                y = line.mean_case_median.to_numpy()
                lower, upper = line.lower_mean_minus_sd.to_numpy(), line.upper_mean_plus_sd.to_numpy()
                ax.fill_between(x, lower, upper,
                                color=COLORS[method], alpha=0.12, linewidth=0)
                ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                        markersize=3.7, markeredgewidth=0.6,
                        linewidth=1.9 if method == "general_adaptive" else 1.45,
                        label=LABELS[method], zorder=3 if method == "general_adaptive" else 2)
            ax.set_xscale("log")
            ax.set_yscale(yscale)
            ax.grid(True, axis="both", which="major", alpha=0.75)
            ax.set_xlim(175, 3450)
            ax.xaxis.set_major_locator(FixedLocator([200, 1000, 3000]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(which="major", length=3, width=0.6, pad=3)
            if row == 0:
                ax.set_title(r"$N = " + format(budget, ",") + r"$", pad=9)
            if col == 0:
                ax.set_ylabel(row_label + "\nError", labelpad=8)
            if row == nrows - 1:
                ax.set_xlabel(r"Dimension $d$", labelpad=5)
    if reference:
        # Finish all data panels before reading shared-axis bounds. Guides must
        # never change the ranges of this panel or another panel in its row.
        bounds = [(ax.get_xlim(), ax.get_ylim()) for ax in axes.flat]
        for ax, (xlim, ylim) in zip(axes.flat, bounds):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_autoscale_on(False)
        for ax, (xlim, ylim) in zip(axes.flat, bounds):
            x_min, x_max = xlim
            y_min, y_max = ylim
            for guide in REFERENCES:
                power = guide["power"]
                x_stop = min(x_max, x_min * (y_max / y_min) ** (1.0 / power))
                x_ref = np.geomspace(x_min, x_stop, 100)
                y_ref = y_min * (x_ref / x_min) ** power
                ax.plot(x_ref, y_ref, color="#242A30", linestyle=guide["style"],
                        linewidth=1.2, label=guide["label"], zorder=1.5)
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], markersize=4,
                      linewidth=1.8, label=LABELS[m]) for m in METHODS]
    if reference:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.115 if nrows == 1 else 0.055),
                   ncol=5, frameon=False, handlelength=1.9, columnspacing=1.2)
        reference_handles = [Line2D([0], [0], color="#242A30", linestyle=guide["style"],
                                    linewidth=1.2, label=guide["label"]) for guide in REFERENCES]
        fig.legend(handles=reference_handles, loc="lower center", bbox_to_anchor=(0.53, 0.009),
                   ncol=3, frameon=False, handlelength=2.6, columnspacing=2.0)
    else:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.012),
                   ncol=len(handles), frameon=False, handlelength=1.9, columnspacing=1.2)
    fig.subplots_adjust(left=0.104, right=0.975, top=0.86 if nrows == 1 else 0.94,
                        bottom=(0.38 if nrows == 1 else 0.17) if reference else 0.135, hspace=0.22, wspace=0.20)
    fig.savefig(output, metadata={"Creator": "Gaussian trial-level figure refresh", "CreationDate": None, "ModDate": None})
    fig.savefig(Path(output).with_suffix('.png'), dpi=160)
    plt.close(fig)

def order_scatter_figure(per_case, out, proxy, *, max_order=6):
    """Show every generator and its embedding mean, with the paper's order limit."""
    plt.rcdefaults()
    if max_order not in range(1, 9):
        raise ValueError("order limit must be between one and eight")
    p=per_case[per_case.proxy.eq(proxy) & per_case.weighting.eq('ordinary_ols') & per_case.sample_schedule.eq('uniform_n') & per_case.num_points.eq(15) & per_case.order.between(1,max_order)].copy()
    assert len(p)==21*max_order and p.groupby(['embedding','order']).size().eq(7).all() and p.trials.eq(10).all()
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(6.5,2.7),sharex=True)
    metrics=[('absolute_center_error','Center error (CE)'),('trial_sd','Trial SD'),('rmse','RMSE')]
    styles=[('fid','Inception','#0072B2','o',-.12),('fd_dinov2','DINOv2','#D55E00','s',0),('clip','CLIP','#009E73','^',.12)]
    positive_values=[[],[],[]]
    rows=[]
    for emb,label,color,marker,offset in styles:
        sub=p[p.embedding.eq(emb)].copy()
        means=sub.groupby('order')[[x[0] for x in metrics]].mean().sort_index()
        orders=means.index.to_numpy()
        for j,(metric,title) in enumerate(metrics):
            values=means[metric].to_numpy()
            assert np.isfinite(values).all() and (values>0).all()
            axes[j].plot(orders,values,color=color,marker=marker,linestyle='-',linewidth=1.3,markersize=3.5,markeredgewidth=0,label=label,zorder=4)
            for order,group in sub.groupby('order'):
                group=group.sort_values('generator')
                x=order+offset+np.linspace(-.045,.045,7)
                y=group[metric].to_numpy()
                assert np.isfinite(y).all() and (y>0).all()
                axes[j].scatter(x,y,s=12,facecolors='none',edgecolors=color,marker=marker,linewidths=.65,alpha=.42,zorder=2)
                positive_values[j].extend(y)
                for (_,row),position in zip(group.iterrows(),x):
                    rows.append({'metric':metric,'embedding':emb,'order':int(order),'generator':row.generator,'value':row[metric],'plot_x':position})
    for j,(ax,(_,title)) in enumerate(zip(axes,metrics)):
        ax.set_yscale('log')
        ax.set_ylim(min(positive_values[j])/1.25,max(positive_values[j])*1.25)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y',which='major',color='#dedede',linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(.75,max_order+.25);ax.set_xticks(range(1,max_order+1))
        ax.tick_params(which='both',direction='out',length=2.5)
        ax.set_title(title,pad=7);ax.set_xlabel(r'Polynomial order $k$',labelpad=3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1.005),handlelength=2.5,columnspacing=1.7)
    fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.75,wspace=.35)
    stem = f"imagenet_ols_order_{proxy}_absolute"
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{stem}.png", dpi=250)
    pd.DataFrame(rows).to_csv(out / "ols_order_scatter_points.csv", index=False)
    assert len(rows) == 63 * max_order
    plt.close(fig)


PROXY_STYLES = {
    'plugin': ('Empirical', '#55616C', 's'),
    'fid_infinity': (r'FID$_\infty$', '#0072B2', '^'),
    'rtd': ('RTD', '#009E73', 'o'),
}


def save_figure(fig, output, dpi=210):
    fig.savefig(output, metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(Path(output).with_suffix('.png'), dpi=dpi)
    plt.close(fig)


def intro_figure(gaussian, imagenet, output):
    styles()
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10})
    methods = {'fid_infinity': PROXY_STYLES['fid_infinity'],
               'vale2': (r'VALE$_2$', '#D55E00', 'v'), 'rtd': PROXY_STYLES['rtd']}
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.15))
    for method, (label, color, marker) in methods.items():
        g = gaussian[gaussian.method.eq(method)].sort_values('dimension')
        r = imagenet[imagenet.method.eq(method)].sort_values('sample_budget')
        for ax, x, y in [(axes[0], g.dimension, g.mean_case_median),
                          (axes[1], r.sample_budget/1000, r.median_estimate)]:
            ax.plot(x, y, color=color, marker=marker, linewidth=1.45,
                    markersize=3.8, markeredgewidth=.5, label=label)
    axes[0].set_title(r'(a) Random Gaussian, $N=50$K', pad=9)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlim(175, 3450)
    axes[0].xaxis.set_major_locator(FixedLocator([200, 1000, 3000]))
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].set_xlabel(r'Dimension $d$'); axes[0].set_ylabel('Absolute error')
    axes[1].set_title('(b) DDO/EDM2-L 512, Inception', pad=9)
    axes[1].set_xlim(40, 310); axes[1].set_xticks([50, 100, 150, 200, 250, 300])
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[1].set_xlabel(r'Sample size $N$ (K)'); axes[1].set_ylabel('FID estimate')
    for ax in axes:
        ax.grid(True, alpha=.8); ax.set_axisbelow(True)
        ax.margins(y=.09); ax.tick_params(length=3, width=.6, pad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.54, .995), columnspacing=2.5, handlelength=2)
    fig.subplots_adjust(left=.095, right=.987, bottom=.185, top=.775, wspace=.33)
    save_figure(fig, output)


def proxy_panel(ax, rows, title):
    for proxy, (label, color, marker) in PROXY_STYLES.items():
        curve = rows[rows.proxy.eq(proxy)].sort_values('sample_budget')
        ax.plot(curve.sample_budget/1000, curve.median_estimate, label=label,
                color=color, marker=marker, linewidth=1.05, markersize=2.7, markeredgewidth=.45)
    ax.set_title(title, pad=4)
    ax.set_xticks([30, 100, 200, 300]); ax.set_xlim(18, 312)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = ScalarFormatter(useOffset=False); formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(length=2.4, pad=2.2)
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(alpha=.20, linewidth=.45); ax.margins(y=.1)


def proxy_figures(trend, out):
    """One three-embedding example and three six-generator appendix sheets."""
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.3,
                         'axes.titlesize': 7.8, 'axes.linewidth': .65,
                         'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    embeddings = [('fid', 'inception', 'Inception'), ('fd_dinov2', 'dinov2', 'DINOv2'),
                  ('clip', 'clip', 'CLIP')]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    example = trend[trend.generator.eq('stylegan_xl_imagenet256')]
    for ax, (embedding, _, label) in zip(axes, embeddings):
        proxy_panel(ax, example[example.embedding.eq(embedding)], label)
        ax.set_ylabel('FID estimate' if embedding == 'fid' else 'FD estimate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .995),
               ncol=3, frameon=False, fontsize=8)
    fig.supxlabel(r'Sample size $N$ (K)', fontsize=8, x=.55, y=.025)
    fig.subplots_adjust(left=.085, right=.99, bottom=.24, top=.73, wspace=.48)
    save_figure(fig, out/'imagenet_proxy_example.pdf')

    generators = [('stylegan_xl_imagenet64', 'StyleGAN-XL (64)'),
                  ('ddo_edm2_s_imagenet64', 'DDO/EDM2-S (64)'),
                  ('var_d30_imagenet256', 'VAR-d30 (256)'),
                  ('stylegan_xl_imagenet512', 'StyleGAN-XL (512)'),
                  ('ddo_edm2_l_imagenet512', 'DDO/EDM2-L (512)'),
                  ('var_d36_imagenet512', 'VAR-d36 (512)')]
    for embedding, name, _ in embeddings:
        fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.12), sharex=True)
        fig.subplots_adjust(left=.10, right=.988, bottom=.155, top=.82, wspace=.36, hspace=.40)
        for ax, (generator, label) in zip(axes.flat, generators):
            rows = trend[trend.embedding.eq(embedding) & trend.generator.eq(generator)]
            proxy_panel(ax, rows, label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .985),
                   ncol=3, frameon=False, handlelength=2.1, columnspacing=2.4, fontsize=8)
        fig.supxlabel(r'Sample size $N$ (K)', fontsize=8, x=.55, y=.025)
        fig.supylabel('FID estimate' if embedding == 'fid' else 'FD estimate',
                      fontsize=8, x=.012, y=.49)
        save_figure(fig, out/f'imagenet_proxy_consistency_{name}_additional.pdf')
