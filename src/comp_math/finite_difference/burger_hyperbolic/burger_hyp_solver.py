import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class KIRSolver:
    # Схема Куранта-Изаксона-Риса (КИР)

    def __init__(self, x, u0, dt, t_end, form='non_div', bc_type='periodic'):
        self.x = x
        self.dx = x[1] - x[0]
        self.Nx = len(x)
        self.u0 = u0.copy()
        self.dt = dt
        self.t_end = t_end
        self.Nt = int(np.ceil(t_end / dt))
        self.form = form
        self.bc_type = bc_type
        
        max_u = np.max(np.abs(u0))
        courant = max_u * dt / self.dx
        if courant >= 1:
            print("Предупреждение: число куранта >= 1")
    
    def apply_boundary_conditions(self, u):
        if self.bc_type == 'periodic':
            u[0] = u[-2]
            u[-1] = u[1]
        else:
            u[0] = u[1]
            u[-1] = u[-2]
        return u
    
    def solve(self, verbose=True):
        u = self.u0.copy()
        history = [u.copy()]
        
        for n in range(self.Nt):
            u_new = u.copy()
            
            for i in range(1, self.Nx - 1):
                if self.form == 'non_div':
                    if u[i] >= 0:
                        flux = u[i] * (u[i] - u[i-1]) / self.dx
                    else:
                        flux = u[i] * (u[i+1] - u[i]) / self.dx
                else:
                    if u[i] >= 0:
                        flux = ((u[i]**2)/2 - (u[i-1]**2)/2) / self.dx
                    else:
                        flux = ((u[i+1]**2)/2 - (u[i]**2)/2) / self.dx
                
                u_new[i] = u[i] - self.dt * flux
            
            u_new = self.apply_boundary_conditions(u_new)
            u = u_new
            history.append(u.copy())
            
            if verbose and (n+1) % max(1, self.Nt // 10) == 0:
                print(f"  t = {(n+1)*self.dt:.4f}")
        
        self.history = np.array(history)
        return self.history
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.x, self.u0, 'b--', label='t=0', lw=2, alpha=0.7)
        ax.plot(self.x, self.history[-1], 'r-', label=f't={self.t_end}', lw=2)
        
        # Показываем несколько промежуточных моментов
        n_steps = len(self.history)
        for step in [n_steps//4, n_steps//2, 3*n_steps//4]:
            if step < n_steps:
                t = step * self.dt
                alpha = 0.3
                ax.plot(self.x, self.history[step], 'gray', alpha=alpha, linewidth=1)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u', fontsize=12)
        ax.set_title(f'КИР схема (1-й порядок)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()


class LaxWendroffSolver:
    # Схема Лакса-Вендроффа

    def __init__(self, x, u0, dt, t_end, form='non_div', bc_type='periodic', viscosity=0.01):
        self.x = x
        self.dx = x[1] - x[0]
        self.Nx = len(x)
        self.u0 = u0.copy()
        self.dt = dt
        self.t_end = t_end
        self.Nt = int(np.ceil(t_end / dt))
        self.form = form
        self.bc_type = bc_type
        self.viscosity = viscosity
        
        max_u = np.max(np.abs(u0))
        courant = max_u * dt / self.dx
    
    def apply_boundary_conditions(self, u):
        if self.bc_type == 'periodic':
            u[0] = u[-2]
            u[-1] = u[1]
        else:
            u[0] = u[1]
            u[-1] = u[-2]
        return u
    
    def add_viscosity(self, u):
        """Добавление искусственной вязкости для подавления осцилляций"""
        u_new = u.copy()
        for i in range(1, self.Nx - 1):
            # Лапласиан для вязкости
            laplacian = (u[i+1] - 2*u[i] + u[i-1]) / (self.dx**2)
            u_new[i] += self.viscosity * self.dt * laplacian
        return u_new
    
    def solve(self, verbose=True):
        u = self.u0.copy()
        history = [u.copy()]
        
        for n in range(self.Nt):
            u_new = u.copy()
            
            # Предиктор
            u_half = np.zeros(self.Nx - 1)
            for i in range(self.Nx - 1):
                if self.form == 'non_div':
                    u_avg = 0.5 * (u[i] + u[i+1])
                    u_half[i] = 0.5 * (u[i] + u[i+1]) - 0.5 * self.dt/self.dx * u_avg * (u[i+1] - u[i])
                else:
                    u_half[i] = 0.5 * (u[i] + u[i+1]) - 0.5 * self.dt/self.dx * ((u[i+1]**2)/2 - (u[i]**2)/2)
            
            # Корректор
            for i in range(1, self.Nx - 1):
                if self.form == 'non_div':
                    u_avg = 0.5 * (u_half[i-1] + u_half[i])
                    flux = u_avg * (u_half[i] - u_half[i-1]) / self.dx
                else:
                    flux = ((u_half[i]**2)/2 - (u_half[i-1]**2)/2) / self.dx
                
                u_new[i] = u[i] - self.dt * flux
            
            # Добавляем искусственную вязкость
            u_new = self.add_viscosity(u_new)
            
            u_new = self.apply_boundary_conditions(u_new)
            u = u_new
            history.append(u.copy())
            
            if verbose and (n+1) % max(1, self.Nt // 10) == 0:
                print(f"  t = {(n+1)*self.dt:.4f}")
        
        self.history = np.array(history)
        return self.history
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.x, self.u0, 'b--', label='t=0', lw=2, alpha=0.7)
        ax.plot(self.x, self.history[-1], 'r-', label=f't={self.t_end}', lw=2)
        
        n_steps = len(self.history)
        for step in [n_steps//4, n_steps//2, 3*n_steps//4]:
            if step < n_steps:
                t = step * self.dt
                ax.plot(self.x, self.history[step], 'gray', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u', fontsize=12)
        ax.set_title(f'Лакс-Вендрофф схема (2-й порядок, вязкость={self.viscosity})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()


class WKLSolver:
    # Схема Уорминга-Кутлера-Ломакса
    
    def __init__(self, x, u0, dt, t_end, form='non_div', bc_type='periodic'):
        self.x = x
        self.dx = x[1] - x[0]
        self.Nx = len(x)
        self.u0 = u0.copy()
        self.dt = dt
        self.t_end = t_end
        self.Nt = int(np.ceil(t_end / dt))
        self.form = form
        self.bc_type = bc_type
        
        max_u = np.max(np.abs(u0))
        courant = max_u * dt / self.dx
    
    def apply_boundary_conditions(self, u):
        if self.bc_type == 'periodic':
            u[0] = u[-4]
            u[1] = u[-3]
            u[-2] = u[2]
            u[-1] = u[3]
        else:
            u[0] = u[1]
            u[1] = u[2]
            u[-2] = u[-3]
            u[-1] = u[-2]
        return u
    
    def solve(self, verbose=True):
        u = self.u0.copy()
        history = [u.copy()]
        
        kir_solver = KIRSolver(self.x, u, self.dt, self.dt, self.form, self.bc_type)
        u = kir_solver.solve(verbose=False)[1]
        history.append(u.copy())
        
        for n in range(2, self.Nt + 1):
            u_new = u.copy()
            
            for i in range(2, self.Nx - 2):
                if self.form == 'non_div':
                    if u[i] >= 0:
                        flux = u[i] * (1.5*(u[i] - u[i-1]) - 0.5*(u[i-1] - u[i-2])) / self.dx
                    else:
                        flux = u[i] * (1.5*(u[i+1] - u[i]) - 0.5*(u[i+2] - u[i+1])) / self.dx
                else:
                    if u[i] >= 0:
                        flux = (1.5*((u[i]**2)/2 - (u[i-1]**2)/2) - 
                               0.5*((u[i-1]**2)/2 - (u[i-2]**2)/2)) / self.dx
                    else:
                        flux = (1.5*((u[i+1]**2)/2 - (u[i]**2)/2) - 
                               0.5*((u[i+2]**2)/2 - (u[i+1]**2)/2)) / self.dx
                
                u_new[i] = u[i] - self.dt * flux
            
            u_new = self.apply_boundary_conditions(u_new)
            u = u_new
            history.append(u.copy())
            
            if verbose and n % max(1, self.Nt // 10) == 0:
                print(f"  t = {n*self.dt:.4f}")
        
        self.history = np.array(history[:self.Nt+1])
        return self.history
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.x, self.u0, 'b--', label='t=0', lw=2, alpha=0.7)
        ax.plot(self.x, self.history[-1], 'r-', label=f't={self.t_end}', lw=2)
        
        n_steps = len(self.history)
        for step in [n_steps//4, n_steps//2, 3*n_steps//4]:
            if step < n_steps:
                t = step * self.dt
                ax.plot(self.x, self.history[step], 'gray', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u', fontsize=12)
        ax.set_title(f'WKL схема (2-й порядок upwind)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
