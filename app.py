import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "replace-with-a-secret-key"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ML model
MODEL_PATH = "student_model.pkl"
model = pickle.load(open(MODEL_PATH, "rb"))

# Max marks mapping (as provided)
MAX_MARKS = {
    "AL501": 70,
    "AL502": 70,
    "AL503": 70,
    "AL504": 70,
    "AL505": 30,
    "AL506": 30,
    "AL507": 100,
    "AL508": 50,
    "Attendance": 100,
    "Assignments": 10
}

FEATURE_COLS = ["AL501","AL502","AL503","AL504","AL505","AL506","AL507","AL508","Attendance","Assignments"]
TOTAL_MAX = sum([MAX_MARKS[c] for c in ["AL501","AL502","AL503","AL504","AL505","AL506","AL507","AL508"]])  # 490

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def numeric_series_from_df(df, col, default=0):
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def generate_suggestion(row, pred_label):
    # Identify weak subjects: <40% of that subject's max
    weak = []
    for subj in ["AL501","AL502","AL503","AL504","AL505","AL506","AL507","AL508"]:
        val = row.get(subj, 0)
        maxv = MAX_MARKS[subj]
        if maxv > 0 and (val / maxv) * 100 < 40:
            weak.append(subj)
    # basic suggestions by label
    if pred_label == "Excellent":
        base = "Excellent performance. Maintain consistency and revise weak topics occasionally."
    elif pred_label == "Average":
        base = "Average. Increase study hours on weak subjects and complete all assignments on time."
    else:
        base = "At Risk. Immediately seek help from teachers, focus on fundamentals and increase attendance."
    if weak:
        subj_str = ", ".join(weak)
        base += f" Weak subjects: {subj_str}. Focus practice and doubt-clearing in these areas."
    return base

@app.route("/")
def index():
    return render_template("index.html", max_marks=MAX_MARKS)

@app.route("/predict", methods=["POST"])
def predict():
    # Single student predict (same as old)
    try:
        vals = []
        for f in FEATURE_COLS:
            vals.append(float(request.form.get(f, 0)))
        features = np.array([vals])
        pred = model.predict(features)[0]
        # suggestion
        row = dict(zip(FEATURE_COLS, vals))
        suggestion = generate_suggestion(row, pred)
        return render_template("index.html", max_marks=MAX_MARKS, prediction=f"Predicted Performance: {pred}", suggestion=suggestion)
    except Exception as e:
        return render_template("index.html", max_marks=MAX_MARKS, prediction=f"Error: {str(e)}")

@app.route("/teacher")
def teacher():
    return render_template("teacher_upload.html", max_marks=MAX_MARKS)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    # Handle CSV upload and batch predictions
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("teacher"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("teacher"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
        file.save(save_path)

        # Read CSV into DataFrame
        try:
            df = pd.read_csv(save_path)
        except Exception as e:
            flash(f"Failed to read CSV: {str(e)}")
            return redirect(url_for("teacher"))

        # Required columns
        required_cols = set(["StudentID","Name"] + FEATURE_COLS)
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            flash(f"CSV is missing required columns: {', '.join(missing)}")
            return redirect(url_for("teacher"))

        if len(df) > 100:
            flash("CSV contains more than 100 students. Please upload max 100 rows.")
            return redirect(url_for("teacher"))

        # Ensure numeric and fill NaNs
        for c in FEATURE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Prepare features in right order
        X = df[FEATURE_COLS].values
        preds = model.predict(X)
        proba = model.predict_proba(X)  # requires classifier with proba
        classes = model.classes_.tolist()
        # get indexes if exist
        idx_excellent = classes.index("Excellent") if "Excellent" in classes else None
        idx_atrisk = classes.index("At Risk") if "At Risk" in classes else None

        # compute total marks and normalized score
        subject_cols = ["AL501","AL502","AL503","AL504","AL505","AL506","AL507","AL508"]
        df["total_marks"] = df[subject_cols].sum(axis=1)
        df["normalized_total"] = df["total_marks"] / float(TOTAL_MAX)

        # probability for 'Excellent' if available
        if idx_excellent is not None:
            df["proba_excellent"] = proba[:, idx_excellent]
        else:
            df["proba_excellent"] = 0.0

        # score used for ranking: weighted combination
        df["rank_score"] = df["proba_excellent"] * 0.7 + df["normalized_total"] * 0.3

        # add predicted label
        df["predicted_label"] = preds

        # generate suggestions
        df["suggestion"] = df.apply(lambda r: generate_suggestion(r, r["predicted_label"]), axis=1)

        # Attendance >= 75%
        df["attendance_ok"] = df["Attendance"].apply(lambda x: True if x >= 75 else False)

        # Assignments submitted all?
        max_assign = MAX_MARKS["Assignments"]
        df["assignments_full"] = df["Assignments"].apply(lambda x: True if x >= max_assign else False)

        # Rank top 10 and bottom 10 (by rank_score)
        df_sorted_desc = df.sort_values(by="rank_score", ascending=False).reset_index(drop=True)
        top_10 = df_sorted_desc.head(10)
        bottom_10 = df_sorted_desc.tail(10).sort_values(by="rank_score", ascending=True)

        # Save a results CSV
        result_filename = f"results_{timestamp}_{filename}"
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
        df.to_csv(result_path, index=False)

        # Convert small tables to dicts for template
        top10_records = top_10[["StudentID","Name","total_marks","predicted_label","proba_excellent","suggestion","attendance_ok","assignments_full"]].to_dict(orient="records")
        bottom10_records = bottom_10[["StudentID","Name","total_marks","predicted_label","proba_excellent","suggestion","attendance_ok","assignments_full"]].to_dict(orient="records")
        attendance_list = df[df["attendance_ok"]][["StudentID","Name","Attendance"]].to_dict(orient="records")
        assignments_full_list = df[df["assignments_full"]][["StudentID","Name","Assignments"]].to_dict(orient="records")

        return render_template("teacher_upload.html",
                               max_marks=MAX_MARKS,
                               top10=top10_records,
                               bottom10=bottom10_records,
                               attendance_list=attendance_list,
                               assignments_full_list=assignments_full_list,
                               result_file=result_filename,
                               total_count=len(df))
    else:
        flash("Only CSV files allowed.")
        return redirect(url_for("teacher"))

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
