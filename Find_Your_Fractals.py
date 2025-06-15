#Lets create the newton fractal 
"""So fractals are objects that show the property of self similarity like exact to a part of itself"""
"""the newton's method : lets have a func f(x) and lets take a new starting value x0, now get the tangent line of f(x0)  and we have to choose another x1 and reepeat it iteratively
formula is x(n+1)=xn-(f(xn)/f'(xn))'"""
import numpy as np
import matplotlib.pyplot as plt
import hashlib

# Fractal Type & Color Map Selection 

fractal_types = {
    0: 'mandelbrot',
    1: 'julia',
    2: 'burning_ship',
    3: 'newton',
    4: 'phoenix',
    5: 'lyapunov'
}#at te end we have to make sure that this list will be chosen from hashmapped name

cmap_groups = {
    "warm": ['plasma', 'inferno', 'magma', 'turbo'],
    "cool": ['viridis', 'twilight_shifted', 'ocean', 'terrain'],
    "futuristic": ['cubehelix', 'gnuplot'],
    "natural": ['terrain', 'cividis', 'ocean'],
    "balanced": ['twilight_shifted', 'viridis', 'cividis'],
    "high_contrast": ['turbo', 'inferno', 'gnuplot']
}#cmap means colour map, it is a way to represent the data in a visual way, and so that our name can be a deciding part in slecting the colour for our fractal type
#completed on 13|06
def name_hash(name):
    return int(hashlib.sha256(name.encode()).hexdigest(), 16)
"""the way how it works is that the hashlib.sha256  is that it takes the name as input and converts it into a hash value, which is a fixed-size string of characters that is unique to the input.
This hash value is then converted into an integer using base 16 (hexadecimal) representation, which allows us to use it for indexing or other purposes. The resulting integer can be used to select elements from lists or perform other operations based on the name. 
so for mandel brot and julia set we need a integer constant so we can take it from name itslef but for lyapunov we need hexa dec values so keep this in mind,we will use the name_hash function to get a unique integer value based on the input name, which can then be used to select the fractal type and color map."""

def name_to_complex(name, scale=1.5):
    h = hashlib.sha256(name.encode()).hexdigest()
    #here we are using the sha256 hash function to convert the name into a hexadecimal string, which is then split into two parts: real and imaginary components.
    # The first 8 characters represent the real part, and the next 8 characters represent the imaginary part of the cmplx no.
    #interpret the hexadecimal values as integers, scale them to a range of -scale to +scale, and return a complex number.
    # The scale parameter allows us to adjust the range of the complex number.
    real = int(h[:8], 16)
    imag = int(h[8:16], 16)
    real_scaled = scale * (real / 0xFFFFFFFF - 0.5)
    imag_scaled = scale * (imag / 0xFFFFFFFF - 0.5)
    return complex(real_scaled, imag_scaled)

def get_cmap_from_name(name):
    #this is the function that will return the colour map based on the name
    # We use the hash of the name to select a color map group and then a specific color map within that group.
    h = name_hash(name)
    group_names = list(cmap_groups.keys())
    group = cmap_groups[group_names[h % len(group_names)]]
    return group[h % len(group)]

def get_fractal_type(name):
    return fractal_types[name_hash(name) % len(fractal_types)] 
#completed on 15|06

# Fractal Functions: here we will work with the types of fractals that we have defined above. lessgo
#mabdel brot set
def mandelbrot(c, max_iter):
    #the formula for mandelbrot set is z = z*z + c, where z is a complex number and c is a constant complex number.
    #the mandelbrot set is the set of complex numbers c for which the sequence defined by the above formula does not diverge. our c will be the encoded name
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter
#my fav julia
def julia(z, c, max_iter):
    #well for julia its the same as of mandel brot but here for specific patterns based on constants, like mandelbrot is like the map of julia set
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter
#titanic ship 
def burning_ship(c, max_iter):
    #the burning ship fractal is a variation of the mandelbrot set, where the real and imaginary parts of the complex number are squared separately and then added.
    #the formula is z = abs(z.real)**2 + abs(z.imag)**2 + c, where z is a complex number and c is a constant complex number.
    #this creates a fractal that resembles a burning ship, hence the name. remeber to change the name as titanic at the end
    z = 0
    for i in range(max_iter):
        z = complex(abs(z.real), abs(z.imag))**2 + c
        if abs(z) > 2:
            return i
    return max_iter
# Newton's Fractal: this is a fractal that is generated using the newton's method, which is a root-finding algorithm for finding the roots of a function.
# The formula is z = z - f(x)/f'(x), where f(x) is a function and f'(x) is its derivative.
#the one who hit by apple
def newton(z, max_iter=50):
    #the reason for max_iter is that we want to limit the number of iterations to avoid infinite loops.
    #the function we are using is f(x) = x^3 - 1, which has three roots: 1, -1/2 + 0.866i, and -1/2 - 0.866i. orignally intened to find the root of a eqn using recursion
    def f(x): return x**3 - 1
    def f_prime(x): return 3 * x**2
    for i in range(max_iter):
        try:
            z = z - f(z)/f_prime(z)
        except ZeroDivisionError:
            return i
        if abs(f(z)) < 1e-6:
            return i
    return max_iter
#pheonix
def phoenix(z, c, p, max_iter):
    #the phoenix fractal is a variation of the julia set, where the complex number is iterated using a combination of two constants c and p.
    #the formula is z = z*z + c + p * prev, where z is a complex number, c is a constant complex number, and p is another constant complex number.
    #here for the complex number we will use the name break it into two like we did for mandelbrot and julia and then use it as c and p. 
    #just reverse the name for p faster and beeter than using the name twice
    prev = 0
    for i in range(max_iter):
        z, prev = z*z + c + p * prev, z
        if abs(z) > 2:
            return i
    return max_iter #completed on 14|06
#lyupunox
def lyapunov(sequence, width=600, height=400):
    # The Lyapunov fractal is generated based on a binary sequence of 'A' and 'B' characters.
    # Each character corresponds to a different growth rate in the logistic map. so a logistic map is a mathematical function that describes how populations grow over time.
    # The sequence is used to determine the growth rates for the logistic map, which is then iterated to compute the Lyapunov exponent
    # The Lyapunov exponent measures the average rate of separation of infinitesimally close trajectories, indicating chaos in the system.
    # The sequence is a string of 'A' and 'B' characters, where 'A' corresponds to a growth rate of A and 'B' corresponds to a growth rate of B.
    A_vals = np.linspace(2.5, 4.0, width)
    B_vals = np.linspace(2.5, 4.0, height)
    data = np.zeros((height, width))
    seq_len = len(sequence)
    for y, B in enumerate(B_vals):
        for x, A in enumerate(A_vals):
            r_values = [A if ch == 'A' else B for ch in sequence]
            x_n = 0.5
            lyapunov = 0.0
            for i in range(100):
                r = r_values[i % seq_len]
                x_n = r * x_n * (1 - x_n)
                try:
                    lyapunov += np.log(abs(r * (1 - 2 * x_n)))
                except:
                    lyapunov = -100
                    break
            data[y, x] = lyapunov / 100
    return data#started on 14 to 15|06  

# ---------- Fractal Generator ----------

def generate_fractal(name, width=800, height=800, zoom=1.0, x_offset=0, y_offset=0, max_iter=200):
    c = name_to_complex(name)
    p = name_to_complex(name[::-1])
    cmap = get_cmap_from_name(name)
    ftype = get_fractal_type(name)

    data = np.zeros((height, width))

    if ftype == 'lyapunov':
        # Generate binary A/B sequence from name hash
        hash_bits = bin(name_hash(name))[2:]
        seq = ''.join(['A' if b == '0' else 'B' for b in hash_bits[:10]])
        return lyapunov(seq, width, height), ftype, cmap

    for x in range(width):
        for y in range(height):
            re = (x - width / 2) / (0.5 * zoom * width) + x_offset
            im = (y - height / 2) / (0.5 * zoom * height) + y_offset
            z = complex(re, im)

            if ftype == 'mandelbrot':
                value = mandelbrot(z, max_iter)
            elif ftype == 'julia':
                value = julia(z, c, max_iter)
            elif ftype == 'burning_ship':
                value = burning_ship(z, max_iter)
            elif ftype == 'newton':
                value = newton(z, max_iter)
            elif ftype == 'phoenix':
                value = phoenix(z, c, p, max_iter)
            else:
                value = 0

            data[y, x] = value

    return data, ftype, cmap

# ---------- Display ----------

def display(name):
    print(f"\n🌌 Generating fractal for: {name}")
    data, ftype, cmap = generate_fractal(name)
    print(f"🌀 Fractal type: {ftype}")
    print(f"🎨 Colormap: {cmap}")

    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap, extent=[-2, 2, -2, 2])
    plt.axis('off')
    plt.title(f"{ftype.title()} Fractal for '{name}'", fontsize=12)
    plt.show()

# ---------- Entry ----------


import time
import textwrap

def magical_intro():
    message = """
    🔍 Welcome to the Realm of "Find Your Fractal"

    A soul’s signature, woven in code,  
In fractal form, your pattern unfolds.  

No echo repeats, no twin remains —  
Unless another dares share your name.

    -->Using a hash of your name, I mapped it to a fractal pattern  
    creating astounding visuals that are mathematically personal.

    Thanks to Code in Place, we learned how simple code can unlock infinite beauty,
    and how curiosity turns into creation.

    So don’t stop with just your name —
    try everything. Explore the patterns within.

    That’s what CIP taught us: to keep going, keep coding, and keep discovering.

    🔑 Enter your name to find your Fractal:
    """

    for line in textwrap.dedent(message).strip().split('\n'):
        print(line)
        time.sleep(0.4)
        #completed on 15|06
magical_intro()
user_input = input("Enter your name (or any string): ")
display(user_input)