from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
import differentiable_rendering as dr

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
STATIC_FOLDER = os.path.join(os.getcwd(), "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
posix_static = "static"

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return redirect(request.url)
        filename = secure_filename(file.filename)
        uid = uuid.uuid4().hex
        saved_name = f"{uid}_{filename}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
        file.save(saved_path)

        # run renderer
        out_prefix = os.path.join(posix_static, uid)
        target = dr.load_image(saved_path, size=128)
        res = dr.optimize(
            target,
            shape_type=request.form.get("shape_type", "gaussians"),
            num_shapes=int(request.form.get("num_shapes", 150)),
            iters=int(request.form.get("iters", 300)),
            out_prefix=out_prefix,
            save_gif=True,
        )

        paths = res["paths"]
        # build URLs for template
        uid_base = uid
        result = {
            "uid": uid_base,
            "uploaded_url": (
                url_for("static", filename=saved_name)
                if False
                else
                # we saved upload in uploads/ not static; provide a direct path via uploads
                f"/uploads/{saved_name}"
            ),
            "start_url": url_for("static", filename=f"{uid_base}_start.png"),
            "final_url": url_for("static", filename=f"{uid_base}_final.png"),
            "gif_url": url_for("static", filename=f"{uid_base}.gif"),
            "frames_dir": f"{uid_base}_frames",
            "num_frames": paths.get("num_frames", 0),
            "reference_url": f"/uploads/{saved_name}",
        }
    return render_template("index.html", result=result)


if __name__ == "__main__":
    # expose uploads folder in dev server
    from flask import send_from_directory

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    app.run(debug=True, host="127.0.0.1", port=5000)
