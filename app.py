from flask import Flask, render_template
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def index():
    # Your existing code for generating the plot
    plt.figure(figsize=(15,5))
    plt.plot(test_df.index, test_df['Predicted'], label='Predicted')
    plt.plot(test_df.index, test_df['Actual'], label='Actual')
    plt.plot(future_df_fin.index, future_df_fin['Predicted'], label='Future Predicted')
    plt.title('Predicted vs Actual Daily Yield')
    plt.xlabel('Date')
    plt.ylabel('Daily Yield')
    plt.legend()

    # Save the plot as an in-memory image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    buf.close()

    # Render the image in the HTML template
    return render_template("index.html", img_base64=img_base64)

if __name__ == "__main__":
    app.run(debug=True)