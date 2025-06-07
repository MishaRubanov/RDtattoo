#!/usr/bin/env python
# coding: utf-8

# ## Initialize default parameters and simulator

# In[2]:


from rdtattoo import tattoo_plotter as tp
import plotly
from rdtattoo.plotly_colorscales import oslo
from rdtattoo.rd_defaults import grayscott_worm_default
plotly.offline.init_notebook_mode()

sim = grayscott_worm_default

# ### Generate random initial arrays and run the model

# In[2]:


a_initial,

# ### Plot the resulting simulation

# In[ ]:


fig = tp.create_plotly_figure(a2, oslo, 1)
fig.show()

# ### Varying the textures
# We can manipulate the textures by changing A/B

# In[ ]:


rd.alpha = 0.01
rd.beta = 0.25
t, output_a, output_b = rd.run(a, b)
tp.create_plotly_figure(output_a, oslo, 1)

# In[ ]:


rd.alpha = 0.01
rd.beta = 1
t, output_a, output_b = rd.run(a, b)
tp.create_plotly_figure(output_a, oslo, 1)

# In[ ]:


import numpy as np
from scipy.ndimage.interpolation import rotate

def average_rotate(a, degree):
    """
    Takes a 2d array a, and produces the average arrays,
    having rotated it degree times. The resulting shape
    has approximate degree-fold rotational symmetry.
    """
    theta = 360 / degree

    a = np.mean([rotate(a, theta * i, reshape=False) for i in range(degree)], axis=0)

    return a


def random_symmetric_initialiser(shape, degree):
    """
    Random initialiser with degree-fold symmetry.
    """
    
    a = np.random.normal(loc=0, scale=0.05, size=shape)
    b = np.random.normal(loc=0, scale=0.05, size=shape)

    return (
        average_rotate(a, degree), 
        average_rotate(b, degree)
    )

# In[8]:


rx = random_symmetric_initialiser((100,100), 5)

# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(rx[0])

# In[4]:


import rd_simulator_gui as rds

rds.create_max_value_sim(rds.fitzhugh_nagumo_default, 10).model_dump()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
plt.scatter(np.linspace(0, 1, 21), np.logspace(0.1, 10, 21), log=True)
%matplotlib inline



plt.show()

# In[ ]:



