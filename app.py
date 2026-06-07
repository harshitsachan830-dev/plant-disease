from flask import Flask, render_template, request, send_from_directory, jsonify
import tensorflow as tf
import numpy as np
import json
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


from PIL import Image

import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
REPORT_FOLDER = "reports"
app.config["REPORT_FOLDER"] = REPORT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model

model = tf.keras.models.load_model("models/plant_disease_model.keras")
print("✅ Plant Disease Model Loaded Successfully")
print(f"📊 Total Parameters: {model.count_params():,}")
try:
    model.summary(print_fn=lambda x: None)
except Exception:
    pass

HEATMAP_FOLDER = "heatmaps"
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

def generate_gradcam(img_array, model):

    try:

        base_model = model.get_layer("efficientnetb0")

        last_conv_layer = base_model.get_layer("top_conv")

        activation_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        activations = activation_model(img_array)

        heatmap = tf.reduce_mean(
            activations[0],
            axis=-1
        )

        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

        print("Heatmap generated successfully")

        return heatmap.numpy()

    except Exception as e:

        print("Heatmap Error:", e)

        return None


def generate_ai_analysis(disease_name, confidence, description, treatment):
    return f"""
Disease: {disease_name}

Confidence: {confidence}%

Description:
{description}

Treatment:
{treatment}

Recommendation:
Monitor the plant regularly and follow the suggested treatment steps.
""".strip()

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Disease Information
disease_info = {
    "Tomato_healthy": {
        "description": "Healthy tomato plant.",
        "treatment": "No treatment required."
    },

    "Potato___healthy": {
        "description": "Healthy potato plant.",
        "treatment": "No treatment required."
    },

    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant.",
        "treatment": "No treatment required."
    },

    "Tomato_Early_blight": {
        "description": "Fungal disease causing dark spots.",
        "treatment": "Apply fungicide and remove infected leaves."
    },

    "Tomato_Late_blight": {
        "description": "Serious fungal infection.",
        "treatment": "Apply copper fungicide immediately."
    },

    "Tomato_Bacterial_spot": {
        "description": "Bacterial infection causing leaf spots.",
        "treatment": "Use copper-based bactericide."
    },

    "Tomato_Leaf_Mold": {
        "description": "Fungal disease in humid conditions.",
        "treatment": "Improve ventilation and use fungicide."
    },

    "Tomato_Septoria_leaf_spot": {
        "description": "Small circular spots on leaves.",
        "treatment": "Remove infected leaves and spray fungicide."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Mite infestation causing yellow leaves.",
        "treatment": "Use neem oil or insecticidal soap."
    },

    "Tomato__Target_Spot": {
        "description": "Fungal disease causing concentric lesions.",
        "treatment": "Use recommended fungicides."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral infection spread by whiteflies.",
        "treatment": "Control whiteflies and remove infected plants."
    },

    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing leaf discoloration.",
        "treatment": "Remove infected plants immediately."
    },

    "Potato___Early_blight": {
        "description": "Fungal disease affecting potato leaves.",
        "treatment": "Apply fungicide and rotate crops."
    },

    "Potato___Late_blight": {
        "description": "Rapidly spreading potato disease.",
        "treatment": "Use fungicide and destroy infected plants."
    },

    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial leaf disease in peppers.",
        "treatment": "Apply copper sprays and avoid overhead watering."
    }
}


@app.route("/", methods=["GET", "POST"])
def index():
    top_predictions = []
    plant_name = None
    disease_name = None
    prediction_status = None
    severity = None
    health_score = None
    confidence_color = "secondary"
    severity_class = "secondary"

    prediction = None
    confidence = None
    description = None
    treatment = None
    image_path = None
    report_path = None
    heatmap_path = None
    ai_analysis = None

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                prediction="Invalid File",
                confidence=0,
                description="Only JPG, JPEG and PNG images are allowed.",
                treatment="Please upload a plant leaf image.",
                image_path=None,
                top_predictions=[],
                report_path=None,
                prediction_status="Invalid File",
                severity="N/A",
                plant_name=None,
                disease_name=None,
            )

        if file and file.filename != "":

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            # Leaf Validation
            try:
                img_check = Image.open(filepath).convert("RGB")
            except Exception:
                return render_template(
                    "index.html",
                    prediction="Invalid File",
                    confidence=0,
                    description="The uploaded file is not a valid image.",
                    treatment="Please upload JPG, JPEG or PNG images only.",
                    image_path=None,
                    top_predictions=[],
                    report_path=None,
                    prediction_status="Invalid File",
                    severity="N/A",
                    plant_name=None,
                    disease_name=None,
                )
            img_np = np.array(img_check)

            green_pixels = np.sum(
                (img_np[:, :, 1] > img_np[:, :, 0]) &
                (img_np[:, :, 1] > img_np[:, :, 2])
            )

            total_pixels = img_np.shape[0] * img_np.shape[1]
            green_ratio = green_pixels / total_pixels

            if green_ratio < 0.15:

                prediction = "Not a Plant Leaf"
                confidence = 0
                description = (
                    "The uploaded image does not appear to be a plant leaf."
                )
                treatment = (
                    "Please upload a clear image of a tomato, potato, or pepper leaf."
                )

                image_path = f"/uploads/{filename}"

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    description=description,
                    treatment=treatment,
                    image_path=image_path,
                    top_predictions=[],
                    report_path=None,
                    prediction_status="Invalid Image",
                    severity="N/A",
                    plant_name=None,
                    disease_name=None,
                )

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)

            img_array = tf.keras.applications.efficientnet.preprocess_input(
                img_array
            )

            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)

            heatmap_path = None

            heatmap = generate_gradcam(img_array, model)

            if heatmap is not None:

                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original = cv2.imread(filepath)
                original = cv2.resize(original, (224, 224))

                superimposed = cv2.addWeighted(
                    original,
                    0.6,
                    heatmap,
                    0.4,
                    0
                )

                heatmap_filename = "heatmap_" + filename

                heatmap_file_path = os.path.join(
                    HEATMAP_FOLDER,
                    heatmap_filename
                )

                cv2.imwrite(
                    heatmap_file_path,
                    superimposed
                )

                heatmap_path = f"/heatmaps/{heatmap_filename}"

            # Top 3 predictions
            top_indices = np.argsort(preds[0])[-3:][::-1]

            top_predictions = []


            for idx in top_indices:
                top_predictions.append({
                    "name": class_names[idx],
                    "confidence": round(float(preds[0][idx]) * 100, 2)
                })

            class_index = top_indices[0]

            prediction = class_names[class_index]
            confidence = round(float(preds[0][class_index]) * 100, 2)

            # Reject unsupported leaves (mango, neem, tulsi, random plants, etc.)
            second_confidence = round(float(preds[0][top_indices[1]]) * 100, 2)

            allowed_plants = ["Tomato", "Potato", "Pepper"]
            predicted_plant = prediction.split("_")[0]

            if (
                predicted_plant not in allowed_plants
                or confidence < 90
                or (confidence - second_confidence) < 30
            ):

                prediction = "Unsupported Leaf"
                description = (
                    "This model supports only Tomato, Potato and Pepper leaves. "
                    "The uploaded image appears to belong to another plant or the prediction is unreliable."
                )
                treatment = (
                    "Upload a clear Tomato, Potato or Pepper leaf image."
                )

                image_path = f"/uploads/{filename}"

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    description=description,
                    treatment=treatment,
                    image_path=image_path,
                    top_predictions=top_predictions,
                    report_path=None,
                    prediction_status="Unsupported Leaf",
                    severity="N/A",
                    plant_name=None,
                    disease_name=None,
                    heatmap_path=heatmap_path,
                    ai_analysis="""
The uploaded image does not appear to be a supported crop leaf.

Supported crops:
• Tomato
• Potato
• Pepper

The model was trained only on these crops, therefore predictions for Mango, Tulsi, Neem, Rose and other plants may be inaccurate.

Recommendation:
Upload a clear image of a Tomato, Potato or Pepper leaf for reliable disease detection.
"""
                )
            # Prediction Status
            if confidence >= 90:
                prediction_status = "Very High Confidence"
            elif confidence >= 75:
                prediction_status = "High Confidence"
            elif confidence >= 60:
                prediction_status = "Medium Confidence"
            else:
                prediction_status = "Low Confidence"

            if confidence >= 90:
                confidence_color = "success"
            elif confidence >= 70:
                confidence_color = "warning"
            else:
                confidence_color = "danger"

            # Severity Level
            if confidence >= 95:
                severity = "High 🔴"
                severity_class = "danger"
                health_score = 40
            elif confidence >= 85:
                severity = "Medium 🟠"
                severity_class = "warning"
                health_score = 65
            else:
                severity = "Low 🟢"
                severity_class = "success"
                health_score = 85

            if "healthy" in prediction.lower():
                severity = "Healthy 🟢"
                severity_class = "success"
                health_score = 100

            plant_name = prediction.split("_")[0]
            disease_name = prediction.replace(plant_name, "")
            disease_name = disease_name.replace("_", " ").strip()

            

            # Reject uncertain predictions
            if confidence < 70:

                prediction = "Uncertain Prediction"

                description = (
                    "The model is not confident enough to identify the disease."
                )

                treatment = (
                    "Please upload a clearer image focused on a tomato, potato, or pepper leaf."
                )

                ai_analysis = generate_ai_analysis(
                    top_predictions[0]['name'],
                    confidence,
                    description,
                    treatment
                )

            else:

                info = disease_info.get(
                    prediction,
                    {
                        "description": "No description available.",
                        "treatment": "No treatment available."
                    }
                )

                description = info["description"]
                treatment = info["treatment"]

                ai_analysis = generate_ai_analysis(
                    disease_name,
                    confidence,
                    description,
                    treatment
                )

            image_path = f"/uploads/{filename}"

            pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
            pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)

            doc = SimpleDocTemplate(pdf_path)
            styles = getSampleStyleSheet()

            content = [
                Paragraph("Plant Disease Detection Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"<b>Plant:</b> {plant_name}", styles["Normal"]),
                Paragraph(f"<b>Disease:</b> {disease_name}", styles["Normal"]),
                Paragraph(f"<b>Confidence:</b> {confidence}%", styles["Normal"]),
                Paragraph(f"<b>Severity:</b> {severity}", styles["Normal"]),
                Paragraph(f"<b>Health Score:</b> {health_score}/100", styles["Normal"]),
                Spacer(1, 12),
                Paragraph(f"<b>Description:</b> {description}", styles["Normal"]),
                Spacer(1, 12),
                Paragraph(f"<b>Treatment:</b> {treatment}", styles["Normal"]),
                Spacer(1, 12),
                Paragraph(f"<b>AI Analysis:</b> {ai_analysis if ai_analysis else 'Not Available'}", styles["Normal"]),
            ]

            doc.build(content)
            report_path = f"/reports/{pdf_filename}"

            created_at = datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )

            conn = sqlite3.connect("plant_history.db")
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO predictions
                (plant, disease, confidence, status, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    plant_name,
                    prediction,
                    confidence,
                    prediction_status,
                    severity,
                    created_at,
                )
            )

            conn.commit()
            conn.close()
    return render_template(
        "index.html",
        prediction=prediction,
        plant_name=plant_name if prediction and prediction != "Uncertain Prediction" else None,
        disease_name=disease_name if prediction and prediction != "Uncertain Prediction" else None,
        confidence=confidence,
        description=description,
        treatment=treatment,
        image_path=image_path,
        top_predictions=top_predictions,
        report_path=report_path,
        prediction_status=prediction_status,
        severity=severity,
        health_score=health_score,
        confidence_color=confidence_color,
        severity_class=severity_class,
        heatmap_path=heatmap_path,
        ai_analysis=ai_analysis,
    )




# History page route
@app.route("/history")
def history():
    conn = sqlite3.connect("plant_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT plant, disease, confidence, status, severity
        FROM predictions
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return render_template(
        "history.html",
        predictions=rows
    )


# Dashboard route
@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("plant_history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE disease LIKE '%healthy%'
    """)
    healthy_count = cursor.fetchone()[0]

    diseased_count = total_predictions - healthy_count

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        WHERE disease != 'Uncertain Prediction'
        GROUP BY disease
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    most_common_disease = result[0] if result else "N/A"
    if most_common_disease != "N/A":
        most_common_disease = most_common_disease.replace("___", " ")
        most_common_disease = most_common_disease.replace("__", " ")
        most_common_disease = most_common_disease.replace("_", " ")

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = round(cursor.fetchone()[0] or 0, 2)

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        WHERE disease NOT LIKE '%healthy%'
          AND disease != 'Uncertain Prediction'
        GROUP BY disease
        ORDER BY COUNT(*) DESC
        LIMIT 5
    """)

    disease_stats = cursor.fetchall()

    cursor.execute("""
        SELECT plant, disease, confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """)
    recent_predictions = cursor.fetchall()

    cursor.execute("""
    SELECT substr(created_at,1,10),
    COUNT(*)
    FROM predictions
    WHERE created_at IS NOT NULL
    GROUP BY substr(created_at,1,10)
    ORDER BY substr(created_at,1,10)
    LIMIT 7
    """)

    trend_data = cursor.fetchall()

    os.makedirs("static/charts", exist_ok=True)

    # Pie Chart
    labels = ["Healthy", "Diseased"]
    sizes = [healthy_count, diseased_count]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Healthy vs Diseased Plants")

    pie_chart_path = "static/charts/pie_chart.png"
    plt.savefig(pie_chart_path)
    plt.close()

    # Bar Chart
    disease_names = []
    disease_counts = []

    for disease, count in disease_stats:
        disease_names.append(
            disease.replace("___", " ").replace("__", " ").replace("_", " ")
        )
        disease_counts.append(count)

    plt.figure(figsize=(8, 5))
    plt.bar(disease_names, disease_counts)
    plt.title("Top Diseases")
    plt.xticks(rotation=20)

    bar_chart_path = "static/charts/bar_chart.png"

    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    trend_dates = []
    trend_counts = []

    for day, count in trend_data:
        if day is not None:
            trend_dates.append(str(day))
            trend_counts.append(count)

    if trend_dates:
        plt.figure(figsize=(8, 4))
        plt.plot(trend_dates, trend_counts, marker="o")
        plt.title("Prediction Trend")
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No trend data available", ha="center")
        plt.title("Prediction Trend")

    trend_chart_path = "static/charts/trend_chart.png"

    plt.tight_layout()
    plt.savefig(trend_chart_path)
    plt.close()

    conn.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        most_common_disease=most_common_disease,
        avg_confidence=avg_confidence,
        recent_predictions=recent_predictions,
        disease_stats=disease_stats,
        pie_chart=pie_chart_path,
        bar_chart=bar_chart_path,
        trend_chart=trend_chart_path,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )



# --- API Endpoints ---

@app.route("/api/stats")
def api_stats():
    conn = sqlite3.connect("plant_history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = round(cursor.fetchone()[0] or 0, 2)

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        GROUP BY disease
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    most_common_disease = result[0] if result else "N/A"

    conn.close()

    return jsonify({
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence,
        "most_common_disease": most_common_disease
    })


@app.route("/api/history")
def api_history():
    conn = sqlite3.connect("plant_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT plant, disease, confidence, status, severity
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    data = []

    for row in rows:
        data.append({
            "plant": row[0],
            "disease": row[1],
            "confidence": row[2],
            "status": row[3],
            "severity": row[4]
        })

    return jsonify(data)


@app.route("/reports/<filename>")
def download_report(filename):
    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )

@app.route("/heatmaps/<filename>")
def heatmap_file(filename):
    return send_from_directory(
        HEATMAP_FOLDER,
        filename
    )

if __name__ == "__main__":
    app.run(debug=True)