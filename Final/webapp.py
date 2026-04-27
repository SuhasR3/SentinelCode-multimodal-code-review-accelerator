from __future__ import annotations

from flask import Flask, render_template, request

from src.final_user_input import analyze_text


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    code = request.form.get("code", "").strip()
    alpha = float(request.form.get("alpha", "0.5"))
    explain_with_llm = request.form.get("explain_with_llm") == "on"

    if not code:
        return render_template(
            "index.html",
            code=code,
            alpha=alpha,
            explain_with_llm=explain_with_llm,
            error="Paste a code snippet before running analysis.",
        )

    try:
        analysis = analyze_text(
            text=code,
            alpha=alpha,
            explain_with_llm=explain_with_llm,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            code=code,
            alpha=alpha,
            explain_with_llm=explain_with_llm,
            error=str(exc),
        )

    return render_template(
        "index.html",
        code=code,
        alpha=alpha,
        explain_with_llm=explain_with_llm,
        analysis=analysis,
    )


if __name__ == "__main__":
    app.run(debug=True)
