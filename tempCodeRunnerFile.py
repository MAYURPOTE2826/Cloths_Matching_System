from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)


model = joblib.load("model.pkl")
df = pd.read_csv("color_dataset.csv")


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

  
    color_aliases = {
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

    for alias, base in color_aliases.items():
        if alias in color_name:
            return base


    return color_name


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        shirt_color = normalize_color_name(request.form["shirt_color"])

        if shirt_color not in color_map:
            return render_template("index.html",
                                   result=f"❌ '{shirt_color}' not recognized. Try colors like blue, red, white, or black.")

        shirt_rgb = np.array(color_map[shirt_color]).reshape(1, -1)

        # Find the closest match from dataset
        df["shirt_rgb"] = list(zip(df["shirt_r"], df["shirt_g"], df["shirt_b"]))
        df["distance"] = df["shirt_rgb"].apply(lambda x: euclidean_distances([x], shirt_rgb)[0][0])
        best = df.loc[df["distance"].idxmin()]

        result = {
            "shirt": shirt_color.capitalize(),
            "pant": best["pant_color_name"].capitalize(),
            "accessory": best["accessory_suggestion"],
            "style": best["style_type"].capitalize()
        }
        return render_template("index.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
