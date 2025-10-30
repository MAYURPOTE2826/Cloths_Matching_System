from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load ML model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv("color_dataset.csv")

# Color reference map
color_map = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "gray": (128, 128, 128),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "maroon": (128, 0, 0),
    "navy": (0, 0, 128),
    "beige": (245, 245, 220),
    "brown": (139, 69, 19),
    "olive": (128, 128, 0)
}


def normalize_color_name(color_name):
    color_name = color_name.lower().strip()

    modifiers = ["dark", "light", "faint", "deep", "pale", "soft", "bright", "off"]
    for mod in modifiers:
        if color_name.startswith(mod + " "):
            color_name = color_name.replace(mod + " ", "")

    aliases = {
        "navy blue": "blue",
        "sky blue": "blue",
        "baby blue": "blue",
        "off white": "white",
        "charcoal black": "black",
        "forest green": "green",
        "lime green": "green",
        "cream": "beige",
        "tan": "brown",
        "golden": "yellow",
        "silver": "gray",
        "maroon red": "maroon",
        "wine red": "maroon"
    }

    for alias, base in aliases.items():
        if alias in color_name:
            return base

    return color_name


def extract_dominant_color(image_file):
    """AI-based color extraction using KMeans clustering."""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((150, 150))
    img_np = np.array(image)
    img_np = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(img_np)
    colors = kmeans.cluster_centers_

    counts = np.bincount(kmeans.labels_)
    dominant_color = colors[np.argmax(counts)]
    return tuple(map(int, dominant_color))


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        shirt_color = request.form.get("shirt_color")
        image_file = request.files.get("shirt_image")

        # If image uploaded → detect color
        if image_file and image_file.filename != "":
            rgb = extract_dominant_color(image_file)
            shirt_rgb = np.array(rgb).reshape(1, -1)
            shirt_color = f"Detected RGB: {rgb}"
        else:
            shirt_color = normalize_color_name(shirt_color)
            if shirt_color not in color_map:
                return render_template("index.html", result=f"❌ '{shirt_color}' not recognized.")
            shirt_rgb = np.array(color_map[shirt_color]).reshape(1, -1)

        # Find best match
        df["shirt_rgb"] = list(zip(df["shirt_r"], df["shirt_g"], df["shirt_b"]))
        df["distance"] = df["shirt_rgb"].apply(lambda x: euclidean_distances([x], shirt_rgb)[0][0])
        best = df.loc[df["distance"].idxmin()]

        result = {
            "shirt": shirt_color,
            "pant": best["pant_color_name"].capitalize(),
            "accessory": best["accessory_suggestion"],
            "style": best["style_type"].capitalize(),
            "shirt_rgb": tuple(map(int, shirt_rgb.flatten())),
            "pant_rgb": (int(best["pant_r"]), int(best["pant_g"]), int(best["pant_b"]))
        }
        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
