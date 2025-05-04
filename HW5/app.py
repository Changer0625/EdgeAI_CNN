from flask import Flask, render_template, request, jsonify
import sqlite3

app = Flask(__name__)

# 初始狀態
elevator_status = {
    "door": "CLOSE",
    "inside": 2,
    "outside": 3
}

@app.route('/')
def index():
    return render_template('index.html',
                           door=elevator_status["door"],
                           inside=elevator_status["inside"],
                           outside=elevator_status["outside"])

@app.route('/get_data')
def update():
    conn = sqlite3.connect('people_counting.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM people_counting ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row:
        return jsonify({
            "external_people": row[1],
            "internal_people": row[2],
            "door_status": row[3],
            "timestamp": row[4]
        })
    else:
        return jsonify({"error": "No data found"})

if __name__ == '__main__':
    app.run(debug=True)