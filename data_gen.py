import numpy as np 
from typing import Union, Tuple, Dict, Optional, Any, List, Callable, Literal 
from pydantic import BaseModel, Field, model_validator
 
def run_autonomous_ode(
        params: BaseModel, 
    ) -> np.ndarray:
    """
    Update the trajectory for a dynamical system.
    """
    current_state = np.array(params.initial_cond, dtype=np.float64)
    if current_state.ndim == 0: # It's a single number (scalar)
        current_state = current_state.reshape(1) # Reshape it to be a 1D array with one element     
    trajectory = np.empty((params.steps + 1, len(params.initial_cond)), dtype=np.float64)
    trajectory[0] = current_state

    for step in range(1, params.steps + 1):
        if params.method == 'euler':
            derivs = params.calc_diff(current_state) 
            current_state = current_state + derivs * params.dt
        elif params.method == 'rk4':
            k1 = params.calc_diff(current_state                    )
            k2 = params.calc_diff(current_state + (k1 * params.dt / 2.0)  )
            k3 = params.calc_diff(current_state + (k2 * params.dt / 2.0)  )
            k4 = params.calc_diff(current_state + (k3 * params.dt)        ) 
            increment =  (k1 + 2*k2 + 2*k3 + k4) * (params.dt / 6.0)
            current_state = current_state + increment
        
        if np.isnan(current_state).any() or np.isinf(current_state).any():
            print(f"Numerical instability at step {step}: state={current_state}") 
            raise ValueError("Numerical instability detected.")
        
        trajectory[step] = current_state
    trajectory.astype(np.float32)
    return trajectory

def run_nonautonomous_ode(
        params: BaseModel, 
    ) -> np.ndarray:
    """
    Update the trajectory for a dynamical system.
    """
    t = 0.0
    current_state = np.array(params.initial_cond, dtype=np.float64)
    trajectory = np.empty((params.steps + 1, len(params.initial_cond)), dtype=np.float64)
    trajectory[0] = current_state
    
    for step in range(1, params.steps + 1):
        if params.method == 'euler':
            derivs  = params.calc_diff(current_state, t) 
            current_state = current_state + derivs * params.dt
            
        elif params.method == 'rk4':    
            k1 = params.calc_diff(current_state,             t,          )
            k2 = params.calc_diff(current_state + k1*params.dt/2.0, t + params.dt/2.0, )
            k3 = params.calc_diff(current_state + k2*params.dt/2.0, t + params.dt/2.0, )
            k4 = params.calc_diff(current_state + k3*params.dt,     t + params.dt,     )

            increment = (k1 + 2*k2 + 2*k3 + k4) * (params.dt / 6.0)
            current_state = current_state + increment

        t = t + params.dt

        if np.isnan(current_state).any() or np.isinf(current_state).any():
            print(f"Numerical instability at step {step}: state={current_state}, t={t}")
            raise ValueError("Numerical instability detected.")

        trajectory[step] = current_state 
    return trajectory

class LorenzParams(BaseModel):
    
    sigma: float = Field(10.0, gt=0, description="Prandtl number, related to fluid viscosity (must be > 0)")
    rho: float = Field(28.0, gt=0, description="Rayleigh number, related to temperature gradient (must be > 0)")
    beta: float = Field(8.0 / 3, gt=0, description="Geometric constant (must be > 0)")
    initial_cond: List[float] = [1.0, 0.98, 1.1]

    dt: float = Field(..., gt=0.0, description="Simulation time increment. Must be > 0.")
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'

    def calc_diff(self, state: np.ndarray) -> np.ndarray: 
        x, y, z = state[0], state[1], state[2]
        d_state = np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ], dtype=np.float64) 
        return d_state
    
    def generate(self) -> np.ndarray: 
        print(f"-> Generating Lorenz trajectory with params: rho={self.rho}")  
        trajectory = run_autonomous_ode(self) 
        return trajectory 


class RosslerParams(BaseModel): 
    a: float = Field(0.2, ge=0.0, description="Linear feedback in y; typically ≥ 0")
    b: float = Field(0.2, ge=0.0, description="Offset in z; typically ≥ 0")
    c: float = Field(5.7, gt=0.0, description="Nonlinear chaos parameter; typically > 0")
    initial_cond: List[float] = [1.0, 1.0, 1.0]

    dt: float = Field(..., gt=0.0, description="Simulation time increment. Must be > 0.")
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'

    def calc_diff(self, state: np.ndarray) -> np.ndarray: 
        x, y, z = state[0], state[1], state[2]
        d_state = np.array([
            -y - z,
            x + self.a * y,
            self.b + z * (x - self.c)
        ], dtype=np.float64) 
        return d_state

    def generate(self) -> np.ndarray:  
        trajectory = run_autonomous_ode(self)
        return trajectory

class HyperRosslerParams(BaseModel):
    a: float = Field(0.25, ge=0.0, description="Linear feedback in y; typically ≥ 0")
    b: float = Field(3.0, ge=0.0, description="Offset in z; typically ≥ 0")
    c: float = Field(0.5, gt=0.0, description="Coupling strength from z to w; must be > 0")
    d: float = Field(0.05, gt=0.0, description="Growth term for w; must be > 0")
    initial_cond: List[float] = [1.0, 1.0, 4.0, 1.0]

    dt: float = Field(..., gt=0.0, description="Simulation time increment. Must be > 0.")
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'

    def calc_diff(self, state: np.ndarray) -> np.ndarray:
        x, y, z, w = state[0], state[1], state[2], state[3]
        dx = -y - z
        dy = x + self.a * y + w
        dz = self.b + x * z
        dw = -self.c * z + self.d * w
        return np.array([dx, dy, dz, dw], dtype=np.float64)
    
    def generate(self) -> np.ndarray:
        trajectory = run_autonomous_ode(self)
        return trajectory


class DuffingParams(BaseModel):
    alpha: float = Field(1.0, ge=0.0, description="Linear stiffness; typically ≥ 0")
    beta: float = Field(-1.0, description="Nonlinear stiffness; can be < 0 for chaotic behavior")
    delta: float = Field(0.2, gt=0.0, description="Damping coefficient; must be > 0")
    gamma: float = Field(0.3, gt=0.0, description="Amplitude of forcing term; must be > 0")
    omega: float = Field(1.0, gt=0.0, description="Frequency of forcing term; must be > 0")
    initial_cond: List[float] = [0.1, 0.1] 

    dt: float = Field(..., gt=0.0, description="Simulation time increment. Must be > 0.")
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'

    def calc_diff(self, state: np.ndarray, t: np.ndarray) -> np.ndarray:
        x, v = state[0], state[1]
        dx = v
        dv = self.gamma * np.cos(self.omega * t) - self.delta * v - self.alpha * x - self.beta * x**3
        d_state = np.array([dx, dv], dtype=np.float64)
        return d_state
    
    def generate(self) -> np.ndarray: 
        trajectory = run_nonautonomous_ode(self)
        return trajectory

 
class Lorenz96Params(BaseModel):
    dim: int = Field(6, gt=3, description="Number of variables in the system; must be greater than 3")
    forcing: float = Field(8.0, description="Forcing term F; chaos typically occurs when F ≥ 8")
    initial_cond: Optional[List[float]] = Field(None, description="Initial condition of length dim")

    dt: float = Field(..., gt=0.0, description="Simulation time increment. Must be > 0.")
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'


    def calc_diff(self, state: np.ndarray, forcing: float) -> np.ndarray:
        x_roll_p1 = np.roll(state, shift=-1, axis=0)  # x_{i+1}
        x_roll_m1 = np.roll(state, shift=1, axis=0)   # x_{i-1}
        x_roll_m2 = np.roll(state, shift=2, axis=0)   # x_{i-2}
        dxdt = (x_roll_p1 - x_roll_m2) * x_roll_m1 - state + forcing
        return dxdt

    def generate(self) -> np.ndarray: 
        if self.initial_cond is None: 
            current_state = np.full((self.dim,), self.forcing, dtype=np.float64)
            current_state[0] += 0.01 # Small perturbation 
        else:
            current_state = self.initial_cond
 
        trajectory = np.empty((self.steps + 1, self.dim), dtype=np.float64)
        trajectory[0] = current_state

        for step in range(1, self.steps + 1):
            if self.method == 'euler':
                d_state_mul_dt = self.calc_diff(current_state, self.forcing) * self.dt
                current_state = current_state + d_state_mul_dt 
            elif self.method == 'rk4':
                k1 = self.calc_diff(current_state, self.forcing)
                k2 = self.calc_diff(current_state + (k1 * self.dt / 2.0), self.forcing)
                k3 = self.calc_diff(current_state + (k2 * self.dt / 2.0), self.forcing)
                k4 = self.calc_diff(current_state + (k3 * self.dt), self.forcing) 
                d_state_mul_dt =  (k1 + 2*k2 + 2*k3 + k4) * (self.dt / 6.0)
                current_state = current_state + d_state_mul_dt
            
            if np.isnan(current_state).any() or np.isinf(current_state).any():
                print(f"Numerical instability at step {step}: state={current_state}") 
                raise ValueError("Numerical instability detected.")
            
            trajectory[step] = current_state
          
        return trajectory
 
def chua_nonlinearity(x: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """
    Dimensionless Chua diode (3-segment PWL):
      h(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
    Equivalent to the classic piecewise form with slopes m0 (outer) and m1 (inner).
    """
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))


class ChuaParams(BaseModel):
    # Canonical double-scroll region (dimensionless)
    alpha: float = Field(15.6, gt=0.0, description="α > 0")
    beta:  float = Field(28.0, gt=0.0, description="β > 0 (chaos around ~25–51 for classic m0,m1)")
    m0:    float = Field(-8.0/7.0, description="Outer-slope of PWL nonlinearity")
    m1:    float = Field(-5.0/7.0, description="Inner-slope of PWL nonlinearity")

    # Simulation settings
    initial_cond: Optional[List[float]] = Field(None, description="(x0,y0,z0)")
    dt:    float = Field(0.005, gt=0.0, description="Time step")
    steps: int   = Field(20000, ge=1, description="Number of integration steps")
    method: Literal['rk4', 'euler'] = 'rk4'

    @model_validator(mode='after')
    def _check_ic(self):
        if self.initial_cond is not None and len(self.initial_cond) != 3:
            raise ValueError("initial_cond must be length 3 for Chua (x,y,z).")
        return self

    def calc_diff(self, state: np.ndarray) -> np.ndarray:
        """
        Dimensionless Chua ODE:
          dx/dt = α * (y - x - h(x))
          dy/dt = x - y + z
          dz/dt = -β * y
        """
        x, y, z = state
        hx = chua_nonlinearity(x, self.m0, self.m1)
        dx = self.alpha * (y - x - hx)
        dy = x - y + z
        dz = -self.beta * y
        return np.array([dx, dy, dz], dtype=np.float64)

    def generate(self, burn_in: int = 1000) -> np.ndarray:
        """
        Returns trajectory of shape (steps+1, 3). Optional burn-in discarded from the front.
        """
        # init
        if self.initial_cond is None:
            current_state = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        else:
            current_state = np.asarray(self.initial_cond, dtype=np.float64)

        total = self.steps + 1 + burn_in
        traj = np.empty((total, 3), dtype=np.float64)
        traj[0] = current_state

        # integrate
        for t in range(1, total):
            if self.method == 'euler':
                k1 = self.calc_diff(current_state)
                current_state = current_state + self.dt * k1
            else:  # rk4
                k1 = self.calc_diff(current_state)
                k2 = self.calc_diff(current_state + 0.5 * self.dt * k1)
                k3 = self.calc_diff(current_state + 0.5 * self.dt * k2)
                k4 = self.calc_diff(current_state + self.dt * k3)
                current_state = current_state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            if not np.isfinite(current_state).all():
                raise ValueError(f"Numerical instability at step {t}: state={current_state}")
            traj[t] = current_state

        return traj[burn_in:]  # drop burn-in to land on the attractor
    
class HenonParams(BaseModel):
    a: float = Field(1.4, description="Nonlinear coefficient; classic chaos at a = 1.4")
    b: float = Field(0.3, description="Linear coefficient; classic chaos at b = 0.3")
    initial_cond: List[float] = Field([0.0, 0.0], min_items=2, max_items=2,
        description="Initial (x, y) values for the Hénon map")
     
    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'
 
    
    def generate(self) -> np.ndarray:
        current_state = np.array(self.initial_cond, dtype=np.float64) 
        trajectory = np.empty((self.steps + 1, 2), dtype=np.float64)
        trajectory[0] = current_state
         
        x, y = current_state[0], current_state[1]
        for step in range(1, self.steps + 1):
            x_new = 1.0 - self.a * x**2 + y
            y_new = self.b * x 
            x, y = x_new, y_new 
            current_state = np.array([x, y], dtype=np.float64) 
            trajectory[step] = current_state

        return trajectory
  
class LogisticParams(BaseModel):
    r: float = Field(3.9, gt=0, description="Growth rate; chaos typically occurs when r ∈ [3.57, 4.0]")
    initial_cond: float = Field(0.5, ge=0.0, le=1.0, description="Initial value x₀ ∈ [0, 1]") 

    steps: int = Field(..., ge=1, description="Must be at least one step")
    device: Literal['cuda', 'cpu'] = 'cuda'
    method: Literal['rk4', 'euler'] = 'rk4'

    def generate(self) -> np.ndarray:
        x = np.array(self.initial_cond, dtype=np.float64)
        trajectory_x = np.empty(self.steps + 1, dtype=np.float64)
        trajectory_x[0] = x
        for step in range(1, self.steps + 1):
            x = self.r * x * (1.0 - x) # Ensure float arithmetic
            trajectory_x[step] = x
        return np.expand_dims(trajectory_x, axis=-1)

 