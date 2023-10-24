#%%
#import numpy as np
from jax import grad
from jax import random
import jax.numpy as np

from jax.test_util import check_grads


def transform(th,s,t,x):
    R = np.array([[np.cos(th),-np.sin(th)],
                  [np.sin(th), np.cos(th)]])
    return R.T@(x-t)/s

def ellipse(a,b,x):
    return (x[0]/a)**2 + (x[1]/b)**2 - 1.0

def ellipse_transform(a,b,th,s,t,x):
    return ellipse(a,b,transform(th,s,t,x))
    
def softmin(x):
    return -np.log(np.exp(-x).sum(axis=0))

def softmax(x):
    return np.exp(-x)/np.exp(-x).sum(axis=0)

def two_ells(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,x):
    l1 = ellipse_transform(a1,b1,th1,s1,t1,x)
    l2 = ellipse_transform(a2,b2,th2,s2,t2,x)
    return softmin(np.array([l1,l2]))

def random_2():
    global key
    if 'key' not in globals():
        key = random.PRNGKey(0)    
    key, subkey = random.split(key)
    return random.normal(key, (2,))

def random_1():
    global key
    if 'key' not in globals():
        key = random.PRNGKey(0)    
    key, subkey = random.split(key)
    return random.normal(key, ())


if True: # check gradients
    # a1,b1,th1,s1,t1 = 1.0,0.5,0.1,1.0,np.array([1.0,0.0]).T
    # a2,b2,th2,s2,t2 = 1.0,0.5,0.0,1.0,np.array([0.0,1.0]).T
    a1,b1,th1,s1,t1 = random_1(),random_1(),random_1(),random_1(),random_2()
    a2,b2,th2,s2,t2 = random_1(),random_1(),random_1(),random_1(),random_2()
    x = random_2()
    print(x)
    # x = np.array([0.5,0.0]).T
    f=two_ells
    f(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,x)
    gf = grad(f,argnums=(0,1,2,3,4,5,6,7,8,9,10))
    gf(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,x)
    check_grads(f,(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,x),order=1)

#%%
import matplotlib.pyplot as plt
import ipywidgets as widgets
%matplotlib widget

res = 128
ran = 2.0
x = np.linspace(-ran, ran, res)
y = np.linspace(-ran, ran, res)
X, Y = np.meshgrid(x, y)

coords = np.vstack((X.ravel(), Y.ravel()))
a1,b1,th1,s1,t1 = 1.0,0.5,0.0,1.0,np.array([[0.0],[0.0]])
a2,b2,th2,s2,t2 = 1.0,0.5,0.0,1.0,np.array([[0.0],[0.0]])
a1,b1 = 0.55,0.55
#z = two_ells(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2).create(coords).reshape(res,res)
z = two_ells(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,coords).reshape(res,res)


fig, ax = plt.subplots(1, 1, figsize=(5,5))

im = ax.imshow(z,cmap="binary_r",extent=(-ran, ran,-ran, ran))

cs = ax.contour(X,Y,z,levels=(0,1),cmap='viridis')
                
ax.clabel(cs, inline=True, fontsize=10)

fig.tight_layout()
plt.grid(True)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()

@widgets.interact(  a1= widgets.FloatSlider(value=0.5, min=0.1, max=1, step=0.001),
                    b1= widgets.FloatSlider(value=0.5, min=0.1, max=1, step=0.001),
                    th1=widgets.FloatSlider(value=0.0, min=0.0, max=2*np.pi, step=0.001),
                    s1= widgets.FloatSlider(value=1, min=-2, max=2, step=0.001),
                    t1x=widgets.FloatSlider(value=0.0, min=-2, max=2, step=0.001),
                    t1y=widgets.FloatSlider(value=0.0, min=-2, max=2, step=0.001),
                    
)
def update(a1, b1, th1,s1,t1x,t1y):
    global cs
    # z = two_ells(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2).create(coords).reshape(res,res)
    t1 = np.array([[t1x],[t1y]])
    z = two_ells(a1,b1,th1,s1,t1,a2,b2,th2,s2,t2,coords).reshape(res,res)
    im.set_data(z)
    cs.remove()
    cs = ax.contour(X,Y,z,levels=(-0.5,0,1),colors=('red','white', 'red'))
    label_positions = []
    for level in range(len(cs.levels)):
        index = np.argmin(cs.get_paths()[level].vertices[:, 1])
        x, y = cs.get_paths()[level].vertices[index]
        label_positions.append((x, y))

    ax.clabel(cs, inline=True, fontsize=10, manual=label_positions, )
    fig.canvas.draw_idle()

# %%
