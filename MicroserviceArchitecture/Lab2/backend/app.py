from flask import Flask, jsonify, request
import psycopg2
import time

app = Flask(__name__)

def get_db_connection():
    """Подключение к БД по имени сервиса db."""
    return psycopg2.connect(
        host="db",
        database="game_db",
        user="user",
        password="password"
    )

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/scores', methods=['GET'])
def get_scores():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT player_name, score FROM scores ORDER BY score DESC LIMIT 10;')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([{'name': r[0], 'score': r[1]} for r in rows])

@app.route('/api/scores', methods=['POST'])
def add_score():
    data = request.json
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO scores (player_name, score) VALUES (%s, %s);',
        (data['name'], data['score'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'status': 'ok'}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)