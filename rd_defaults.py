# Extracted from https://www.ijcce.ac.ir/article_6365_96f54ec0054f8d47df4af016b668c360.pdf
# Here's another https://www.aliensaint.com/uo/java/rd/
from tattoo_functions import RDSimulator, ReactionType

grayscott_worm_default = RDSimulator(
    Da=0.16,
    Db=0.08,
    alpha=0.05,
    beta=0.065,
    dx=1,
    dt=1,
    width=200,
    height=200,
    steps=30000,
    frames=200,
    reaction_type=ReactionType.GRAY_SCOTT,
)

grayscott_bubble_default = RDSimulator(
    Da=0.16,
    Db=0.08,
    alpha=0.035,
    beta=0.065,
    dx=1,
    dt=1,
    width=200,
    height=200,
    steps=30000,
    frames=200,
    reaction_type=ReactionType.GRAY_SCOTT,
)

grayscott_coral_default = RDSimulator(
    Da=0.16,
    Db=0.08,
    alpha=0.06,
    beta=0.062,
    dx=1,
    dt=1,
    width=200,
    height=200,
    steps=30000,
    frames=200,
    reaction_type=ReactionType.GRAY_SCOTT,
)


fitzhugh_nagumo_default = RDSimulator(
    Da=1,
    Db=100,
    alpha=-0.005,
    beta=10,
    dx=1,
    dt=0.001,
    width=200,
    height=200,
    steps=20000,
    frames=100,
    reaction_type=ReactionType.FITZHUGH_NAGUMO,
)

brusselator_default = RDSimulator(
    Da=1,
    Db=8,
    alpha=2,
    beta=3,
    dx=1,
    dt=0.001,
    width=200,
    height=200,
    steps=20000,
    frames=100,
    reaction_type=ReactionType.BRUSSELATOR,
)

# Define equations for each model
model_descriptions = {
    "Fitzhugh-Nagumo": r"""
### Fitzhugh-Nagumo Equations
The Fitzhugh-Nagumo model is described by:

$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - u^3 - v + \alpha$

$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \beta(u - v)$

Chemical Interpretation:
- $u$ represents the fast variable (membrane potential)
- $v$ represents the slow variable (recovery variable)
- The cubic term $-u^3$ represents the fast sodium channel activation
- The term $-v$ represents the slow potassium channel activation
- $\alpha$ represents the external stimulus
- $\beta$ controls the recovery rate

Where:
- $u$ is the activator (state A)
- $v$ is the inhibitor (state B)
- $D_u, D_v$ are diffusion coefficients
- $\alpha, \beta$ are reaction parameters
""",
    "Gray-Scott (Worm)": r"""
### Gray-Scott Equations (Worm Pattern)
The Gray-Scott model is described by:

$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + \alpha(1-u)$

$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (\alpha + \beta)v$

Chemical Reaction Network:
1. $U + 2V \xrightarrow{k_1} 3V$ (autocatalytic)
2. $V \xrightarrow{k_2} P$ (decay to product)


Where:
- $U$ is the activator (state A)
- $V$ is the inhibitor (state B)
- $P$ is the waste product
- $k_1 = 1$ is the autocatalytic rate
- $k_2 = \alpha + \beta$ is the decay rate
- $\alpha$ controls the feed rate of $U$
- $\beta$ controls the removal rate of $V$

Parameters:
- $D_u, D_v$ are diffusion coefficients
- $\alpha, \beta$ are reaction parameters
""",
    "Gray-Scott (Bubble)": r"""
### Gray-Scott Equations (Bubble Pattern)
The Gray-Scott model is described by:

$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + \alpha(1-u)$

$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (\alpha + \beta)v$

Chemical Reaction Network:
1. $U + 2V \xrightarrow{k_1} 3V$ (autocatalytic)
2. $V \xrightarrow{k_2} P$ (decay to product)


Where:
- $U$ is the activator (state A)
- $V$ is the inhibitor (state B)
- $P$ is the waste product
- $k_1 = 1$ is the autocatalytic rate
- $k_2 = \alpha + \beta$ is the decay rate
- $\alpha$ controls the feed rate of $U$
- $\beta$ controls the removal rate of $V$

Parameters:
- $D_u, D_v$ are diffusion coefficients
- $\alpha, \beta$ are reaction parameters
""",
    "Gray-Scott (Coral)": r"""
### Gray-Scott Equations (Coral Pattern)
The Gray-Scott model is described by:

$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + \alpha(1-u)$

$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (\alpha + \beta)v$

Chemical Reaction Network:
1. $U + 2V \xrightarrow{k_1} 3V$ (autocatalytic)
2. $V \xrightarrow{k_2} P$ (decay to product)


Where:
- $U$ is the activator (state A)
- $V$ is the inhibitor (state B)
- $P$ is the waste product
- $k_1 = 1$ is the autocatalytic rate
- $k_2 = \alpha + \beta$ is the decay rate
- $\alpha$ controls the feed rate of $U$
- $\beta$ controls the removal rate of $V$

Parameters:
- $D_u, D_v$ are diffusion coefficients
- $\alpha, \beta$ are reaction parameters
""",
    "Brusselator": r"""
### Brusselator Equations
The Brusselator model is described by:

$\frac{\partial u}{\partial t} = D_u \nabla^2 u + \alpha - (1 + \beta)u + u^2v$

$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \beta u - u^2v$

Chemical Reaction Network:
1. $A \xrightarrow{\alpha} U$ (constant production of $U$)
2. $U \xrightarrow{\beta} V$ (conversion of $U$ to $V$)
3. $2U + V \xrightarrow{k_1} 3U$ (autocatalytic)
4. $U \xrightarrow{k_2} P$ (decay to product)

Where:
- $U$ is the activator (state A)
- $V$ is the inhibitor (state B)
- $A$ is the constant source
- $P$ is the waste product
- $k_1 = 1$ is the autocatalytic rate
- $k_2 = 1$ is the decay rate
- $\alpha$ controls the feed rate
- $\beta$ controls the conversion rate

Parameters:
- $D_u, D_v$ are diffusion coefficients
- $\alpha, \beta$ are reaction parameters
""",
}
