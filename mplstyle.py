import os

import matplotlib

# --- The Updated Style Content ---
style_content = """
# 1. Text Rendering (LaTeX Mode)
text.usetex : True               # Enable external LaTeX requirement
mathtext.fontset : cm            # Use Computer Modern (TeX lookalike) for math

# 2. Fonts
font.family : DejaVu Sans
font.size : 22

# 3. Axes Formatting
axes.linewidth : 1
axes.grid : False
axes.labelweight : normal
"""


def install_style():
    # 1. Get the Matplotlib configuration directory
    # Usually: /home/user/.config/matplotlib
    config_dir = matplotlib.get_configdir()

    # 2. Define the 'stylelib' subfolder
    style_dir = os.path.join(config_dir, "stylelib")

    # 3. Create directory if it doesn't exist
    os.makedirs(style_dir, exist_ok=True)

    # 4. Define the target file path
    target_path = os.path.join(style_dir, "paws.mplstyle")

    # 5. Write the file
    with open(target_path, "w") as f:
        f.write(style_content)

    print("✅ Success!")
    print(f"Style file installed at: {target_path}")
    print("You can now use: plt.style.use('paws')")


if __name__ == "__main__":
    install_style()
