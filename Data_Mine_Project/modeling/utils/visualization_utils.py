import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Set consistent style for all visualizations with larger text"""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Increase font sizes
    plt.rcParams['font.size'] = 14  # Base font size
    plt.rcParams['axes.titlesize'] = 24  # Title font size
    plt.rcParams['axes.labelsize'] = 20  # Axis label size
    plt.rcParams['xtick.labelsize'] = 16  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = 16  # Legend font size
    plt.rcParams['figure.titlesize'] = 28  # Figure title size
    
    # Increase line widths and marker sizes
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10
    
    # Increase tick length and width
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2

def add_model_labels(fig, model_name, analysis_type):
    """Add consistent labels to model visualizations with larger text"""
    fig.suptitle(f'{model_name}: {analysis_type}', 
                fontsize=28, 
                y=1.05,
                weight='bold')

def format_axis_labels(ax, xlabel, ylabel):
    """Format axis labels with larger text"""
    ax.set_xlabel(xlabel, fontsize=20, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)

def add_annotations(ax, text, xy, xytext=None):
    """Add annotations with larger text"""
    if xytext is None:
        xytext = (5, 5)
    ax.annotate(text,
                xy=xy,
                xytext=xytext,
                textcoords='offset points',
                fontsize=16,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

def create_legend(ax, title=None):
    """Create legend with larger text"""
    if title:
        ax.legend(title=title, fontsize=16, title_fontsize=20)
    else:
        ax.legend(fontsize=16)