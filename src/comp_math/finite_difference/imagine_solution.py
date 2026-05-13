import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from IPython.display import HTML

# ==================== ЦВЕТОВЫЕ СХЕМЫ ====================

def get_custom_colormap(name='thermal'):
    if name == 'thermal':
        colors_list = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
                       '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
        return LinearSegmentedColormap.from_list('thermal', colors_list, N=256)
    elif name == 'jet':
        colors_list = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
                       '#FFFF00', '#FFA500', '#FF0000']
        return LinearSegmentedColormap.from_list('jet', colors_list, N=256)
    elif name == 'coolwarm':
        return plt.cm.coolwarm
    elif name == 'plasma':
        return plt.cm.plasma
    else:
        return plt.cm.hot


# ==================== ВИЗУАЛИЗАЦИЯ 2D (x, t) ====================

def plot_heatmap(u, x, t, title="Распределение температуры", cmap='thermal', ax=None):
    """
    Тепловая карта u(x,t). u имеет форму (len(t), len(x)).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    vmin, vmax = np.min(u), np.max(u)
    cm = get_custom_colormap(cmap)
    
    im = ax.imshow(u, aspect='auto', origin='lower', cmap=cm,
                   extent=[x[0], x[-1], t[0], t[-1]], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)
    
    return ax


def plot_profile(u, x, t, time_indices=None, ax=None):
    """
    Профили u(x) в заданные моменты времени.
    time_indices — список индексов слоёв по t.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if time_indices is None:
        time_indices = np.linspace(0, len(t)-1, min(6, len(t)), dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    for idx, c in zip(time_indices, colors):
        ax.plot(x, u[idx, :], color=c, label=f't = {t[idx]:.3f}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Профили в разные моменты времени')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_evolution(u, x, t, space_indices=None, ax=None):
    """
    Эволюция u(t) в заданных точках пространства.
    space_indices — список индексов точек по x.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if space_indices is None:
        space_indices = np.linspace(0, len(x)-1, min(8, len(x)), dtype=int)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(space_indices)))
    for idx, c in zip(space_indices, colors):
        ax.plot(t, u[:, idx], color=c, label=f'x = {x[idx]:.3f}')
    
    ax.set_xlabel('t')
    ax.set_ylabel('u')
    ax.set_title('Эволюция в разных точках пространства')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_3d_surface(u, x, t, cmap='thermal', figsize=(10, 7)):
    """
    3D-поверхность u(x,t).
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t)
    
    cm = get_custom_colormap(cmap)
    surf = ax.plot_surface(X, T, u, cmap=cm, edgecolor='none', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title('Решение u(x,t)')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='u')
    
    return ax


# ==================== АНИМАЦИЯ ====================

def animate_solution(u, x, t, cmap='thermal', interval=50):
    """
    Анимация эволюции профиля u(x) по времени.
    Возвращает HTML5 video для Jupyter Notebook.
    """
    vmin, vmax = np.min(u), np.max(u)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    
    line, = ax.plot([], [], 'r-', linewidth=2)
    fill = ax.fill_between(x, vmin, vmin, alpha=0.3, color='red')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        line.set_data(x, u[frame, :])
        ax.collections.clear()
        ax.fill_between(x, vmin, u[frame, :], alpha=0.3, color='red')
        time_text.set_text(f't = {t[frame]:.4f}')
        return line, time_text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   init_func=init, interval=interval, 
                                   blit=False, repeat=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ==================== ИНТЕРАКТИВНЫЙ СЛАЙДЕР ====================

def interactive_viewer(u, x, t, cmap='thermal'):
    """
    Интерактивный просмотр с ползунком по времени.
    Работает в Jupyter с %matplotlib widget.
    """
    vmin, vmax = np.min(u), np.max(u)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.2)
    
    # Профиль
    line, = ax1.plot(x, u[0, :], 'r-', linewidth=2)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(vmin - 0.05*(vmax-vmin), vmax + 0.05*(vmax-vmin))
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title(f'Профиль при t = {t[0]:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Тепловая карта
    cm = get_custom_colormap(cmap)
    im = ax2.imshow(u, aspect='auto', origin='lower', cmap=cm,
                    extent=[x[0], x[-1], t[0], t[-1]], vmin=vmin, vmax=vmax)
    time_line = ax2.axhline(y=t[0], color='white', linewidth=2, linestyle='--')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Тепловая карта')
    cbar = plt.colorbar(im, ax=ax2, label='u')
    
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 't индекс', 0, len(t)-1, valinit=0, valfmt='%d', valstep=1)
    
    def update(val):
        idx = int(slider.val)
        line.set_ydata(u[idx, :])
        ax1.collections.clear()
        ax1.fill_between(x, vmin, u[idx, :], alpha=0.3, color='red')
        ax1.set_title(f'Профиль при t = {t[idx]:.4f}')
        time_line.set_ydata([t[idx], t[idx]])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


# ==================== ПОЛНЫЙ ОТЧЁТ ====================

def full_report(u, x, t, cmap='thermal'):
    """
    Показывает всё: профили, эволюцию, тепловую карту, гистограмму.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Тепловая карта
    ax1 = fig.add_subplot(2, 2, 1)
    plot_heatmap(u, x, t, title='Тепловая карта u(x,t)', cmap=cmap, ax=ax1)
    
    # Профили
    ax2 = fig.add_subplot(2, 2, 2)
    time_idx = np.linspace(0, len(t)-1, min(6, len(t)), dtype=int)
    plot_profile(u, x, t, time_indices=time_idx, ax=ax2)
    
    # Эволюция
    ax3 = fig.add_subplot(2, 2, 3)
    space_idx = np.linspace(0, len(x)-1, min(6, len(x)), dtype=int)
    plot_evolution(u, x, t, space_indices=space_idx, ax=ax3)
    
    # Гистограмма
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(u.flatten(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.set_xlabel('u')
    ax4.set_ylabel('Частота')
    ax4.set_title('Распределение значений u')
    ax4.grid(True, alpha=0.3, axis='y')
    stats = f'Min: {np.min(u):.4f}\nMax: {np.max(u):.4f}\nСреднее: {np.mean(u):.4f}'
    ax4.text(0.95, 0.95, stats, transform=ax4.transAxes, fontsize=9,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Анализ решения', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()